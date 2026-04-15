import secrets
import string

def generate_secret_key(length=32):
    # Generowanie klucza używając secrets (bezpieczniejsza metoda)
    key = secrets.token_hex(length)
    print("\nWygenerowany SECRET_KEY:")
    print("-" * 50)
    print(key)
    print("-" * 50)
    print("\nSkopiuj ten klucz do pliku .env jako:")
    print("SECRET_KEY=" + key)
    print("\nUwagi:")
    print("1. Ten klucz jest bezpiecznie wygenerowany i odpowiedni do użycia w produkcji")
    print("2. Zapisz go w bezpiecznym miejscu")
    print("3. Nie udostępniaj go publicznie")
    print("4. Używaj tego samego klucza przy każdym uruchomieniu aplikacji")

if __name__ == "__main__":
    generate_secret_key() 