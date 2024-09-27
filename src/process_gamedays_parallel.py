import subprocess
from concurrent.futures import ThreadPoolExecutor

# Define the range of gamedays from "01" to "07"
gamedays = [f"{i:02}" for i in range(1, 8)]  # Creates ['01', '02', ..., '07']


# Function to run the process_gamedays.py script for a given gameday
def run_gameday(gameday):
    command = f"python src/process_gamedays.py {gameday}"

    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return gameday, result.stdout.decode(), result.stderr.decode()


# Run the scripts in parallel using ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_gameday, gameday) for gameday in gamedays]

    for future in futures:
        gameday, stdout, stderr = future.result()
        print(f"Output for gameday {gameday}: {stdout}")
        if stderr:
            print(f"Error for gameday {gameday}: {stderr}")
