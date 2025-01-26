import torch

# Cargar los state_dict de ambos modelos
model1_path = "Experiments/2025-01-23-18-20-31/best_villacampa_lung_organoid_12_1024_0.0001_noise.pt"
model2_path = "Experiments/2025-01-23-18-20-31/post_test.pth"

state_dict1 = torch.load(model1_path)
state_dict2 = torch.load(model2_path)

# Función para comparar los state_dict
def compare_state_dicts(state1, state2):
    if state1.keys() != state2.keys():
        print("Los modelos tienen diferentes estructuras.")
        return False

    for key in state1.keys():
        if not torch.equal(state1[key], state2[key]):
            print(f"Diferencia encontrada en {key}.")
            return False

    print("Los modelos son idénticos.")
    return True

# Comparar los state_dict
are_identical = compare_state_dicts(state_dict1, state_dict2)

if not are_identical:
    print("Los modelos tienen diferencias.")
else:
    print("Los modelos son exactamente iguales.")
