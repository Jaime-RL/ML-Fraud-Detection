import time

# Declaro una variable abecedario con todas las letras
abc = "abcdefghijklmnopqrstuvwxyz "
frase = "chino lameculos"

resultado = ""

# Recorro la frase y el abecedario
for letra in frase:
    for char in abc:
        print(resultado + char)
        time.sleep(0.05)
        if letra == char:
            resultado += char
            break
