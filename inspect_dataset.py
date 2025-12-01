import sys
import os
import pickle
try:
    # Ensure classes referenced in pickle are importable
    import tasks.drug_task  # noqa: F401
except Exception:
    pass

def summarize(obj):
    print('Type:', type(obj))
    if hasattr(obj, '__dict__'):
        print('__dict__ keys:', list(obj.__dict__.keys()))
    if hasattr(obj, 'drugs'):
        try:
            print('drugs count:', len(getattr(obj, 'drugs')))
        except Exception:
            pass
    if hasattr(obj, 'dataset'):
        ds = getattr(obj, 'dataset')
        if isinstance(ds, dict):
            print('splits:', {k: (len(v) if hasattr(v, '__len__') else None) for k, v in ds.items()})
        else:
            print('dataset type:', type(ds))
    if hasattr(obj, 'known'):
        try:
            print('known count:', len(getattr(obj, 'known')))
        except Exception:
            pass
    if hasattr(obj, '_rep_idx'):
        print('rep_idx:', getattr(obj, '_rep_idx'))
    if hasattr(obj, 'sub_lens'):
        print('sub_lens:', getattr(obj, 'sub_lens'))

def main():
    if len(sys.argv) < 2:
        print('Usage: python inspect_dataset.py <path_to_pkl>')
        sys.exit(1)
    fp = sys.argv[1]
    print('Path:', fp)
    try:
        with open(fp, 'rb') as f:
            obj = pickle.load(f)
        summarize(obj)
        # Print split sizes if available
        if hasattr(obj, 'dataset') and isinstance(obj.dataset, dict):
            splits = {k: (len(v) if hasattr(v, '__len__') else None) for k, v in obj.dataset.items()}
            total = sum([splits.get('tr', 0) or 0, splits.get('va', 0) or 0, splits.get('te', 0) or 0])
            print('split sizes:', splits)
            print('total entries:', total)
    except Exception as e:
        print('Load error:', type(e).__name__, str(e))

if __name__ == '__main__':
    main()
