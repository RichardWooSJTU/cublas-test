for raw in range(32):
	for col in range(32):
		res = (((raw % 8) / 2 * 4 + raw /8)*2+raw%2)*32+col
		print(res, end=" ")
	print("\n")

