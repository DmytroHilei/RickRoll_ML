from pathlib import Path

img_folder = Path(
    r"C:\Users\giley\PycharmProjects\Cascade__Training\Build\bin\Debug"
)

extension = ".exe"

count = 0
for file in img_folder.iterdir():
    if file.is_file() and file.suffix.lower() in extension:
        count += 1
        print(f"{count}. {file.name}")

print(count)
