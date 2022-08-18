import sys
import pandas as pd
from paper_table import preproc

def main():
    tmp = preproc(sys.argv[1], 'tnbc')
    tmp = tmp.reset_index()
    tmp["bs"] = tmp.name.apply(lambda x: x.split("_")[-2])
    tmp["MB"] = tmp.name.apply(lambda x: x.split("_")[-3])
    tmp.groupby(['lr', 'wd'], sort=False)['test_score'].max()
    import pdb; pdb.set_trace()
    tmp = tmp.loc[tmp.groupby(["bs", "MB"])["train_score"].idxmax()]
if __name__ == "__main__":
    main()
