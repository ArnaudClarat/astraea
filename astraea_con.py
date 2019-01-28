# coding=utf-8
from lib.beans import *


def menu():
    print("""
        Faites votre choix d'exercices:
        
        1.  Conversions 10 -> Bases binaires    (1 octet non signé)
        2.  Conversions toutes bases            (1 octet non signé)
        3.  Compléments à 2 (Cà2)               (1 octet)
        4.  Décimal -> Flottant                 (2 octets)
        5.  Flottant -> Décimal                 (2 octets)
        6.  Encodage / Décodage de CRC
        7.  Encodage de Hamming
        8.  Décodage de Hamming
        Q.  Quitter l'application
    """)


def show_exercise(mode):
    s_key = ''
    while s_key != 'Q':
        if mode == '1':
            print("""
                    Conversions décimal vers bases binaires"

                    Pensez à préfixer votre réponse par b, 0, 0x !
                    Indiquez votre réponse sur 8 bits (b.... ....) si elle est en binaire.
                    """)
            exercise = Convert(base_src=10, base_dst=choose_bin_base())
        elif mode == '2':
            print("""
                    Conversions toutes bases

                    Pensez à préfixer votre réponse par b, 0, 0x pour les bases binaires
                    Pensez à suffixer votre réponse par () pour les autres bases
                    Indiquez votre réponse sur 8 bits (b.... ....) si elle est en binaire
                    """)

            exercise = Convert()
        elif mode == '3':
            print("""
                    Complément à 2
                    Pensez à préfixer votre réponse par b, 0, 0x pour les bases binaires
                    Indiquez votre réponse sur 8 bits (b.... ....) si elle est en binaire
                    """)

            exercise = Ca2()
        elif mode == '4':
            print("""
                Conversions Décimal vers Flottants 16 bits
            
                Pensez à préfixer votre réponse par b, 0, 0x pour les bases binaires
                Indiquez votre réponse sur 16 bits: 1 bit de signe, 5 bits d'exposants, 10 bits de mantisse

                Astraea ne gère pas (encore) la génération des cas particuliers.
            """)

            exercise = DecimalToFloat16()
        elif mode == '5':
            print("""
                    Conversions Flottants 16 bits vers Décimal

                    Le flottant est sur 16 bits: 1 bit de signe, 5 bits d'exposants, 10 bits de mantisse
                    Vous pouvez écrire la réponse en décimal, ou en polynome de type
                    ±(2^x+2^(x-1)+…)

                    Astraea ne gère pas (encore) la génération des cas particuliers.
            """)

            exercise = Float16ToDecimal()
        elif mode == '6':
            print("""
                    CRC

                    Pensez à préfixer votre réponse par b, 0, 0x
            """)

            exercise = CrcMessage()
        elif mode == '7':
            print("""
                    Encodage de Hamming

                    Pensez à préfixer votre réponse par b, 0, 0x
            """)
            exercise = HammingMessage(encoded=True)
        elif mode == '8':
            print("""
                            Encodage de Hamming

                            Pensez à préfixer votre réponse par b, 0, 0x
                            """)

            exercise = HammingMessage(encoded=False)
        print(exercise.statement)
        sol = input("Votre réponse ?")
        if isinstance(exercise, Float16ToDecimal):
            if sol == exercise.solution or sol == exercise.polynom:
                print("Bonne réponse !")
        elif isinstance(exercise, HammingMessage) and not exercise.encoded:
            err = input("Position de l'erreur ?")
            if sol == exercise.solution and err == exercise.error_pos:
                print("Bonne réponse !")
            elif clear_string(sol) == clear_string(exercise.solution) and err == exercise.error_pos:
                print("Bonne réponse, mais attention au format demandé…")
        elif sol == exercise.solution:
            print("Bonne réponse !")
        elif clear_string(sol) == clear_string(exercise.solution):
            print("Bonne réponse, mais attention au format demandé…")
        else:
            if isinstance(exercise, HammingMessage) and not exercise.encoded:
                print("Désolé, la bonne réponse était : {}, et la position de l'erreur : {}".format(exercise.solution,
                                                                                                    exercise.error_pos))
            else:
                print("Désolé, la bonne réponse était : " + exercise.solution)
        s_key = input("\nTaper Q pour revenir au menu, ou une autre touche pour afficher un nouvel exercice").upper()
    menu()


def clear():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == '__main__':
    clear()
    print("""
        Astraea - Mode console (Design temporaire)
        
        La version avec interface graphique arrive prochainement !
        Ce code est conçu en moins de 30 minutes, il est certainement optimisable ;)
        Il doit être correctement refactoré et documenté.
        
        Bons exercices !
        Cédric.
        
    """)
    key = ''
    menu()
    while key != 'Q':
        key = input("Votre choix ?").upper()
        if not key == 'Q':
            show_exercise(key)
    exit(0)