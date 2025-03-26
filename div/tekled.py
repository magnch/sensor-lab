# Skriv svaret ditt her
a, b, c, Q = 1200, 6, 1.5, 100
def etterspørselselastisitet(a,b,c,Q):
    P = ((Q-a)/-b)**(1/c)
    dQ = b*c*P**(c-1)
    epsilon = dQ * P / Q
    return epsilon
epsilon = etterspørselselastisitet(a,b,c,Q)
print(epsilon)