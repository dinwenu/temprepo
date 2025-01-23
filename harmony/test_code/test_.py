top = 90
bottom = 20
delta_p = int((top - bottom)/90 + 0.5)
print("delta_p: ", delta_p)
for epoch in range(0, 89):
    p = max(top - epoch*delta_p, bottom)
    print(p)