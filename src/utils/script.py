import jwt
import datetime

# Substitua pela mesma chave secreta configurada no servidor
JWT_SECRET_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c'
JWT_ALGORITHM = 'HS256'

# Crie o payload do token
payload = {
    'sub': 'allan',  # Pode ser qualquer identificação
    'iat': datetime.datetime.utcnow(),
    'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1000) 
}

# Gere o token JWT
token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

# Se estiver usando Python 3.6+, o token será um byte string, então converta para string
if isinstance(token, bytes):
    token = token.decode('utf-8')

print(f"Seu token JWT estático: {token}")
