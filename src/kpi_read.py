import pickle
import matplotlib.pyplot as plt


kpis_path = r"C:\Users\psycl\Documents\GitHub\WGU_CS_Capstone\models\model_kpis.pkl"

def read_kpis(kpis_path):
    with open(kpis_path, 'rb') as file:
        kpis = pickle.load(file)
    return kpis

def print_kpis(kpis):
    print("Model KPIs:")
    print(f"Accuracy: {kpis.get('accuracy', 'N/A')}")
    print(f"Precision: {kpis.get('precision', 'N/A')}")
    print(f"Recall: {kpis.get('recall', 'N/A')}")
    print(f"F1 Score: {kpis.get('f1_score', 'N/A')}")

def plot_kpis(kpis, save_path):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [kpis.get('accuracy', 0), kpis.get('precision', 0), kpis.get('recall', 0), kpis.get('f1_score', 0)]
    
    pastel_colors = ['#AEC6CF', '#77DD77', '#FFB347', '#DEA5A4']
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=pastel_colors)
    plt.ylim(0, 1)
    plt.xlabel('KPIs')
    plt.ylabel('Values')
    plt.title('Model Key Performance Indicators')
    
    for i, value in enumerate(values):
        plt.text(i, value + 0.01, f'{value:.2f}', ha='center')
    
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    kpis = read_kpis(kpis_path)
    print_kpis(kpis)
    
    save_path = r"C:\Users\psycl\Documents\GitHub\WGU_CS_Capstone\models\kpi_bar_chart.png"
    plot_kpis(kpis, save_path)
