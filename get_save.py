import os
from pathlib import Path

from dotenv import load_dotenv

from model.utils.bunny import CDNConnector

load_dotenv()

STORAGE_ZONE = '/pollen'
LOCAL_DIR = Path('saves/')

conn = CDNConnector(os.getenv('BUNNY_API_KEY'), STORAGE_ZONE)

list_files = [
    Path(obj['Path']) / obj['ObjectName']
    for obj in conn.get_storaged_objects('models/')
]


print("Choose a number:")

print(*[f"{i}) {file.stem}" for i, file in enumerate(list_files)], sep='\n')
number = int(input('number: '))

chosen = list_files[number]
print(f'Downloading model {chosen}')
pth = str(chosen.relative_to(STORAGE_ZONE))
conn.get_file(pth, LOCAL_DIR / chosen.name)

print("Done.")
