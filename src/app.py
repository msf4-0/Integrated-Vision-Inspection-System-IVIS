"""[summary]
    """
# --------------------------
# Add sys path for modules
import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), 'lib'))  # ./lib
# --------------------------
def main():
    print("Hi")
    

if __name__ == "__main__":
    sys.exit(main())
