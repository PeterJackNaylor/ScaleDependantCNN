import sys
import pandas as pd
import plotly.express as px

def read(path):
    df = pd.read_csv(path, index_col=0).reset_index(drop=True)
    df["augmentation"] = df.name.apply(lambda x: x.split("_")[-2])
    return df
    
order_x = ["vanilla", "greyscale", "autocontrast", "jittersmall", "jittermed", "jitterlarge", "jitterverylarge", "normal"]
tickvals = [k for k in range(len(order_x))]
def main():
    tmp = read(sys.argv[1])

    fig = px.box(tmp, x="augmentation", y="test_score", color="augmentation",  category_orders={"augmentation": order_x},
    color_discrete_sequence=px.colors.qualitative.G10)
    fig.update_layout(title="Augmentation Strategy vs. Linear Score")
    fig.update_xaxes(title_text='Augmentation Strategy')
    fig.update_yaxes(title_text='Linear score')
    fig.update_xaxes(
        ticktext=["Vanilla", "Greyscale", "Autocontrast", "Color Jitter (S)", "ColorJitter (M)", "ColorJitter (L)", "ColorJitter (XL)", "All"],
        tickvals=tickvals
    )

    fig.write_image("augmentation_exp.png")
    fig.write_html("augmentation_exp.html")

    
if __name__ == "__main__":
    main()
