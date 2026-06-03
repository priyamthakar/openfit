"""Extract drda.R source to understand parameterization."""
import tarfile
from pathlib import Path

tarball = Path('/tmp/drda.tar.gz')
if not tarball.exists():
    tarball = Path('C:/Users/priya/AppData/Local/Temp/drda.tar.gz')

tf = tarfile.open(str(tarball), 'r:gz')

# Extract the main drda.R
for m in tf.getmembers():
    if m.name == 'drda/R/drda.R':
        f = tf.extractfile(m)
        content = f.read().decode('utf-8', errors='replace')
        out = Path('D:/openassay/drda_main.R')
        out.write_text(content)
        print(f"Extracted drda.R: {len(content)} bytes")
        break

tf.close()
