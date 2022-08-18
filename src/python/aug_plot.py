import sys
import pandas as pd
import plotly.express as px

def read(path):
    df = pd.read_csv(path, index_col=0).reset_index(drop=True)
    df["augmentation"] = df.name.apply(lambda x: x.split("_")[-2])
    return df
    

def main():
    tmp = read(sys.argv[1])
    fig = px.box(tmp, x="augmentation", y="test_score")
    fig.write_image("augmentation_exp.png")
    fig.write_html("augmentation_exp.html")

    
if __name__ == "__main__":
    main()
