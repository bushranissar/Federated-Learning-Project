import os
print("Files in models/trained:")
for f in os.listdir("models/trained"):
    p = os.path.join("models/trained", f)
    if os.path.isfile(p):
        print(f"  {f}: {os.path.getsize(p)//1024} KB")