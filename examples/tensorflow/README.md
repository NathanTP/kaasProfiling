# Tensorflow examples

## Data Storage
Some of these models use Keras which will store all downloaded data in ~/.keras
by default. You might not want to store data there (e.g. some machines have
NFS-mounted home directories). It's probably a good idea to set KERAS\_HOME to
somewhere better before running performance tests (e.g.
/scratch/USERNAME/.keras).
