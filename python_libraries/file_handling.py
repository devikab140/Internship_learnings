## .txt
# Modifying - Writing (overwrite file)
with open("sample.txt", "w") as f:
    f.write("This is new content.\n")

# Modifying - Appending (add to end)
with open("sample.txt", "a") as f:
    f.write("This line is appended.\n")

# Visualizing - Line by line
with open("sample.txt", "r") as f:
    for line in f:
        print(line.strip())

# Opening and Reading
with open("sample.txt", "r") as f:
    content = f.read()
print("File Content:\n", content)

#-------------------------------------------------#

from pathlib import Path

# Opening and Reading
content = Path("sample.txt").read_text()
print("File Content:\n", content)

# Modifying - Overwriting
Path("sample.txt").write_text("New content written with pathlib.")

# Visualizing
print(Path("sample.txt").read_text())

