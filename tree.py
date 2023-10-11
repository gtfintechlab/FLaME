import os

def display_tree(path, indent=-2):
    # Increment indent to create visual hierarchy
    indent += 2
    # Ensure path is a directory
    if os.path.isdir(path):
        # Print directory name
        print(f"{' ' * indent}{os.path.basename(path)}/")
        # Recursively call for all items in directory
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            display_tree(item_path, indent)
    else:
        # Print file name
        print(f"{' ' * indent}{os.path.basename(path)}")

# Start tree display from current directory or specified path
if __name__ == "__main__":
    import sys
    path = "." if len(sys.argv) == 1 else sys.argv[1]
    display_tree(path)
