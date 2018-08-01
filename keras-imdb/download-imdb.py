import subprocess 

subprocess.check_output(" curl http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz |tar xz")
subprocess.check_output(" tar xvfz aclImdb_v1.tar.gz " )
