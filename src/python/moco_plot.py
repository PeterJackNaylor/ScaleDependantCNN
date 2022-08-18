import sys
import pandas as pd
from paper_table import preproc
import plotly.express as px
def main():
    tmp = preproc(sys.argv[1], 'tnbc')
    tmp = tmp.reset_index()
    tmp["bs"] = tmp.name.apply(lambda x: x.split("_")[-2])
    tmp["MB"] = tmp.name.apply(lambda x: x.split("_")[-3])

    tmp = tmp.loc[tmp.groupby(["bs", "MB"])["train_score"].idxmax()]
    # bs = list(tmp.bs.unique())
    # mb = list(tmp.MB.unique())
    # for bs_ in bs:
    #     for mb_ in mb:
    #         tmpi = tmp.loc[(tmp.bs == bs_) & (tmp.MB == mb_)]
            
    #         y = []
    #         for k in ks:
    #             tmpi[]
    ks = [5, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
    df = pd.DataFrame()
    df_index = 0
    for i in tmp.index:
        for k in ks:
            df.loc[df_index, 'tag'] = f"({tmp.loc[i, 'bs']}, {tmp.loc[i, 'MB']})"
            df.loc[df_index, 'k'] = k
            df.loc[df_index, 'score'] = tmp.loc[i, f'knnscoreK={k}']
            df_index += 1
    
    fig = px.line(df, x="k", y="score", color='tag')
    fig.write_image("scores.png")



if __name__ == "__main__":
    main()
