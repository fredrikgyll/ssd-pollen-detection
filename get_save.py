import subprocess

HOST = 'simple-hybel'
REMOTE_DIR = '/mnt/ssd/saves/'
LOCAL_DIR = 'saves/'
MODEL_NAME = 'ssd_last'
MODEL_EXT = 'pth'

command = subprocess.run(['ssh', HOST, f'ls {REMOTE_DIR}'], capture_output=True)

list_files = command.stdout.decode().split()

print("Choose a number:")

print(*[f"{i}) {file}" for i, file in enumerate(list_files)], sep='\n')
number = int(input('number: '))

chosen = list_files[number]
print(f'Downloading model {chosen}')

command = subprocess.run(
    [
        'scp',
        f'{HOST}:{REMOTE_DIR}{chosen}/{MODEL_NAME}.{MODEL_EXT}',
        f'{LOCAL_DIR}{chosen}.{MODEL_EXT}',
    ]
)

print("Done.")
