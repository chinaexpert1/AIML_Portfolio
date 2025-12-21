    
    import os
    import glob
    
    print("Script started.")
    
    filepaths = glob.glob("storage/*.txt.clean")
    
    print(f"Found {len(filepaths)} files.")
    
    for i, path in enumerate(filepaths):
        print(f"[{i+1}] Reading: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                print(f"Length: {len(text)}")
        except Exception as e:
            print(f"Exception on file {path}: {e}")
    
    print("Finished reading all files.")

    