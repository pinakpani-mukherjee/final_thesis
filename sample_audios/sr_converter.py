import os
import sys

inwave = sys.argv[1]
outwave = sys.argv[2]

output_str = f"ffmpeg -i {inwave} -ac 1 -ar 16000 {outwave}"
os.system(outwave)
print(outwave)