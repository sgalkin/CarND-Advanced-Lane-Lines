import pickle

MATRIX = 'matrix'
DISTORTION = 'distortion'
            
def store(filename, Mc, dist):
    with open(filename, 'wb') as f:
        pickle.dump({MATRIX: Mc, DISTORTION: dist}, f)

def load(filename):
    with open(filename, 'rb') as f:
        bundle = pickle.load(f)
        return bundle[MATRIX], bundle[DISTORTION]
    
if __name__ == "__main__":
    import os
    from tempfile import NamedTemporaryFile
    
    name = None
    with NamedTemporaryFile() as f:
        name = f.name
        M = 'aaa'
        d = 'bbb'
        store(f.name, M, d)
        Mr, dr = load(f.name)
        assert(Mr == M)
        assert(dr == d)
    assert (not os.path.exists(name))
