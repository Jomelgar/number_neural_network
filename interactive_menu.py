import json
import os
import math
import matplotlib.pyplot as plt

#Para imprimir la funcion
def printFigure(images,results,labels, fails = []):
    num_images = len(fails)
    if num_images == 0:
        print("No hay imágenes para mostrar.")
        return
    
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    
    plt.figure(figsize=(cols * 1.5, rows * 1.5))
    for i in range(num_images): 
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(images[fails[i]].reshape(28,28), cmap='gray')

        pred = results[fails[i]]
        real = labels[fails[i]]
        ax.set_title(f"Pred: {pred}\nReal: {real}", 
                     fontsize=6)
        ax.axis("off")

        if i < cols:
            ax.set_title(f"Pred: {pred}\nReal: {real}", fontsize=6, pad=12)
        else:
            ax.set_title(f"Pred: {pred}\nReal: {real}", fontsize=6,pad=6)

    plt.tight_layout(pad=0.5)
    plt.show()


# Menu para trabajar y mostrar todo en sí
class Menu:
    def __init__(self, dict,images,results,labels):
        self.dictionary = dict
        self.images = images
        self.results = results
        self.labels = labels

    def save_to_json(self,data, filename="results.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data,f,indent=4)
        print(f"Resultados guardados en {filename}")

    def show_fails_by_digit(self,digit):
        fails = self.dictionary[f"{digit}"]["fails"]
        if len(fails) == 0:
            print(f"No hay errores para el dígito {digit}")
        else:
            print(f"Mostrando {len(fails)} errores del dígito {digit}")
            printFigure(self.images,self.results,self.labels,fails)

    def show_all_fails(self):
        fails = self.dictionary["fails"]
        if len(fails) == 0:
            print("¡No hay errores!")
        else:
            print(f"Mostrando {len(fails)} errores totales")
            printFigure(self.images,self.results,self.labels,fails)

    def show_statistics(self):
        print("\n" + "="*50)
        print(f"PRECISIÓN GENERAL: {self.dictionary['precision']:.2%} | ERROR: {len(self.dictionary["fails"])}")
        print("="*50)
        for i in range(10):
            precision = self.dictionary[f"{i}"]["precision"]
            fails = len(self.dictionary[f"{i}"]["fails"])
            print(f"Dígito {i}: Precisión {precision:.2%} | Errores: {fails}")
        print("="*50 + "\n")

    def show(self):
        while True:
            print("\n" + "="*50)
            print("MENÚ DE OPCIONES")
            print("="*50)
            print("1. Mostrar estadísticas de precisión")
            print("2. Mostrar todos los errores")
            print("3. Mostrar errores de un dígito específico")
            print("4. Guardar resultados en JSON")
            print("5. Salir")
            print("="*50)

            opcion = input("Selecciona una opción (1-5): ").strip()

            if opcion == "1":
                self.show_statistics()
                os.system("pause")
            elif opcion == "2":
                self.show_all_fails()
                os.system("pause")
            elif opcion == "3":
                try:
                    digit = int(input("Ingresa el dígito (0-9): "))
                    if 0 <= digit <= 9:
                        self.show_fails_by_digit(digit)
                    else:
                        print("El dígito debe estar entre 0 y 9")
                except ValueError:
                    print("Por favor ingresa un número válido")
                os.system("pause")
            elif opcion == "4":
                filename = input("Ingresa el nombre del archivo (default: results.json): ").strip()
                if filename == "":
                    self.save_to_json(self.dictionary)
                else:
                    self.save_to_json(self.dictionary, filename)
                os.system("pause")
            elif opcion == "5":
                print("¡Hasta luego!")
                break
            else:
                print("Opción no válida. Intenta de nuevo.")
                os.system("pause")