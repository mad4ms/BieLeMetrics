from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from difflib import get_close_matches, SequenceMatcher


def _norm(s: str) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .strip()
        .lower()
        .replace("-", " ")
        .replace(".", " ")
        .replace(",", " ")
        .replace("  ", " ")
    )


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _mode_or_first(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return None
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]


@dataclass
class PlayerLeagueMapper:
    """
    Helper to backfill players.league_id by matching Sportradar players to
    Kinexon player identities (names + teams) using exact + fuzzy matching.
    """

    con: "duckdb.DuckDBPyConnection"
    logger: "logging.Logger"

    def build_kinexon_per_name(self) -> pd.DataFrame:
        df_kx_all = self.con.execute(
            """
            SELECT 
                "full name"  AS full_name_kinexon, 
                "group name" AS group_name_kinexon, 
                "league id"  AS league_id, 
                session_id, 
                fixtureId
            FROM kinexon_positions
            WHERE "full name" IS NOT NULL 
              AND "league id" IS NOT NULL
              AND "group name" IS NOT NULL
            """
        ).fetch_df()

        if df_kx_all.empty:
            self.logger.warning(
                "No Kinexon player rows found in kinexon_positions."
            )
            return df_kx_all

        df_kx_all["league_id"] = df_kx_all["league_id"].astype(str)
        df_kx_all["group_name_kinexon"] = df_kx_all[
            "group_name_kinexon"
        ].astype(str)

        kx_ids_per_name = (
            df_kx_all.groupby("full_name_kinexon")["league_id"]
            .nunique()
            .rename("unique_league_ids_per_name")
            .reset_index()
        )

        kx_per_name = (
            df_kx_all.groupby("full_name_kinexon")
            .agg(
                {
                    "league_id": _mode_or_first,
                    "group_name_kinexon": _mode_or_first,
                }
            )
            .reset_index()
            .rename(columns={"league_id": "kin_league_id"})
        )

        kx_per_name = kx_per_name.merge(
            kx_ids_per_name, on="full_name_kinexon", how="left"
        )

        kx_per_name["norm_key"] = kx_per_name["full_name_kinexon"].map(_norm)
        kx_per_name["kin_group_norm"] = kx_per_name["group_name_kinexon"].map(
            _norm
        )

        self.logger.info(
            "KINEXON: distinct player names: %d", len(kx_per_name)
        )
        return kx_per_name

    def load_sportradar_players(self) -> pd.DataFrame:
        df_sr_players = self.con.execute(
            """
            SELECT DISTINCT 
                p.personId, 
                p.nameFullLocal, 
                p.nameFullLatin, 
                p.teamName
            FROM players AS p
            JOIN match_events AS me 
              ON p.personId = me.personId
            """
        ).fetch_df()

        if df_sr_players.empty:
            self.logger.warning(
                "No Sportradar players referenced in match_events."
            )
            return df_sr_players

        df_sr_players["nameFullLocal"] = df_sr_players["nameFullLocal"].astype(
            str
        )
        df_sr_players["teamName"] = df_sr_players["teamName"].astype(str)
        df_sr_players["name_local_norm"] = df_sr_players["nameFullLocal"].map(
            _norm
        )
        df_sr_players["team_norm"] = df_sr_players["teamName"].map(_norm)

        self.logger.info(
            "SPORTRADAR: players referenced in match_events: %d",
            len(df_sr_players),
        )
        return df_sr_players

    def match_players(
        self, kx_per_name: pd.DataFrame, df_sr_players: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (df_match_all, df_map_safe)

        df_match_all: exact + fuzzy matches (one row per personId, first hit wins)
        df_map_safe:  subset restricted to team-aligned matches, used for updates
        """
        if kx_per_name.empty or df_sr_players.empty:
            return pd.DataFrame(), pd.DataFrame()

        kin_map_key_to_name = dict(
            zip(kx_per_name["norm_key"], kx_per_name["full_name_kinexon"])
        )
        kin_map_key_to_league = dict(
            zip(kx_per_name["norm_key"], kx_per_name["kin_league_id"])
        )
        kin_map_key_to_team = dict(
            zip(kx_per_name["norm_key"], kx_per_name["kin_group_norm"])
        )
        kin_keys = list(kin_map_key_to_name.keys())

        rows_exact, rows_fuzzy = [], []

        for _, r in df_sr_players.iterrows():
            pid = r["personId"]
            sr_local = r["nameFullLocal"]
            sr_team = r["teamName"]
            key_local = r["name_local_norm"]
            sr_team_norm = r["team_norm"]

            matched_key = None

            # Exact: name AND team must match
            if (
                key_local in kin_keys
                and kin_map_key_to_team.get(key_local) == sr_team_norm
            ):
                matched_key = key_local

            if matched_key:
                rows_exact.append(
                    (
                        pid,
                        sr_local,
                        sr_team,
                        kin_map_key_to_name[matched_key],
                        kin_map_key_to_league[matched_key],
                        kin_map_key_to_team[matched_key],
                        1.0,
                    )
                )
                continue

            # Fuzzy: restrict search to same team
            eligible_keys = [
                kk
                for kk in kin_keys
                if kin_map_key_to_team.get(kk) == sr_team_norm
            ]

            if not eligible_keys:
                continue

            cand = get_close_matches(
                key_local, eligible_keys, n=1, cutoff=0.84
            )
            if cand:
                best_key = cand[0]
                score = _similarity(key_local, best_key)
                rows_fuzzy.append(
                    (
                        pid,
                        sr_local,
                        sr_team,
                        kin_map_key_to_name[best_key],
                        kin_map_key_to_league[best_key],
                        kin_map_key_to_team[best_key],
                        score,
                    )
                )

        self.logger.info(
            "Exact matches: %d | Fuzzy matches: %d",
            len(rows_exact),
            len(rows_fuzzy),
        )

        df_match_all = pd.DataFrame(
            rows_exact + rows_fuzzy,
            columns=[
                "personId",
                "nameFullLocal_sportradar",
                "teamName_sportradar",
                "full_name_kinexon",
                "league_id",
                "group_name_kinexon_norm",
                "similarity",
            ],
        ).drop_duplicates(subset=["personId"], keep="first")

        df_map_safe = df_match_all.copy()
        df_map_safe["team_norm_sportradar"] = df_map_safe[
            "teamName_sportradar"
        ].map(_norm)

        df_map_safe = df_map_safe.loc[
            df_map_safe["team_norm_sportradar"]
            == df_map_safe["group_name_kinexon_norm"],
            [
                "personId",
                "league_id",
                "teamName_sportradar",
                "team_norm_sportradar",
            ],
        ].drop_duplicates(subset=["personId"], keep="first")

        self.logger.info(
            "Team-aligned mappings to write: %d (skipped due to mismatch: %d)",
            len(df_map_safe),
            max(len(df_match_all) - len(df_map_safe), 0),
        )

        return df_match_all, df_map_safe

    def apply_updates(self, df_map_safe: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the mapping into the players table and returns the updated
        players subset (league_id IS NOT NULL).
        """
        if df_map_safe.empty:
            self.logger.warning(
                "No safe mappings to apply to players.league_id."
            )
            return pd.DataFrame()

        # Reset league_id column
        self.con.execute(
            """ALTER TABLE players DROP COLUMN IF EXISTS league_id"""
        )
        self.con.execute(
            """ALTER TABLE players ADD COLUMN IF NOT EXISTS league_id TEXT"""
        )

        self.con.register("tmp_player_league_map_all", df_map_safe)

        self.con.execute(
            """
            UPDATE players AS p
            SET league_id = tlm.league_id
            FROM tmp_player_league_map_all AS tlm
            WHERE p.personId = tlm.personId
              AND p.teamName = tlm.teamName_sportradar
            """
        )

        self.con.unregister("tmp_player_league_map_all")

        df_players_updated = self.con.execute(
            """
            SELECT *
            FROM players
            WHERE league_id IS NOT NULL
            ORDER BY teamName, nameFullLocal
            """
        ).fetch_df()

        self.logger.info(
            "Updated players.league_id for %d players (team-aligned).",
            df_map_safe["personId"].nunique(),
        )
        return df_players_updated

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline:
          - derive Kinexon per-name map
          - load Sportradar players
          - compute matches and safe mappings
          - apply updates to DuckDB players table

        Returns (df_match_all, df_map_safe, df_players_updated).
        """
        kx_per_name = self.build_kinexon_per_name()
        df_sr_players = self.load_sportradar_players()

        df_match_all, df_map_safe = self.match_players(
            kx_per_name, df_sr_players
        )
        df_players_updated = self.apply_updates(df_map_safe)

        return df_match_all, df_map_safe, df_players_updated
