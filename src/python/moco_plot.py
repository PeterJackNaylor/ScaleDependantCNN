import sys
import pandas as pd
import numpy as np

from paper_table import preproc
import plotly.express as px
def main():
    data = 'tnbc'
    tmp = preproc(sys.argv[1], data)
    tmp = tmp.reset_index()
    tmp["bs"] = tmp.name.apply(lambda x: x.split("_")[-2])
    tmp["MB"] = tmp.name.apply(lambda x: x.split("_")[-3])
    for var in ["test_score", "knnscoreK=40"]:
        import pdb; pdb.set_trace()
        tmp[var + "_std"] = np.array([
            f"{tmp[var].values[i] * 100:.1f}"
            + " \pm "
            + f"{tmp[var + '_std'].values[i] * 100:.1f}"
        for i in range(len(tmp[var + '_std'].values))])

    tmp = tmp[
        [
            "name",
            "bs",
            "MB",
            "lr",
            "wd",
            "test_score_std",
            "knnscoreK=40_std",
        ]
    ]
    tmp.columns = [
        "name",
        "BS",
        "MB",
        "LR",
        "WD",
        "Linear ({})".format(data),
        "knn ({})".format(data),
    ]
    tmp = tmp.drop("name", axis=1)
    tmp["BS"] = tmp["BS"].astype(int)
    tmp["MB"] = tmp["MB"].astype(int)
    tmp = tmp.sort_values(["MB", "BS", "LR", "WD"])
    import pdb; pdb.set_trace()
    tmp = tmp.set_index(["MB", "BS", "LR", "WD"])
    # tmp = tmp.loc[tmp.groupby(["bs", "MB"])["train_score"].idxmax()]
    # # bs = list(tmp.bs.unique())
    # # mb = list(tmp.MB.unique())
    # # for bs_ in bs:
    # #     for mb_ in mb:
    # #         tmpi = tmp.loc[(tmp.bs == bs_) & (tmp.MB == mb_)]
            
    # #         y = []
    # #         for k in ks:
    # #             tmpi[]
    # ks = [40]
    # df = pd.DataFrame()
    # df_index = 0
    # for i in tmp.index:
    #     for k in ks:
    #         df.loc[df_index, 'tag'] = f"({tmp.loc[i, 'bs']}, {tmp.loc[i, 'MB']})"
    #         df.loc[df_index, 'k'] = k
    #         df.loc[df_index, 'score'] = tmp.loc[i, f'knnscoreK={k}']
    #         df_index += 1
    
    # fig = px.line(df, x="k", y="score", color='tag')
    # fig.write_image("scores.png")



if __name__ == "__main__":
    main()
