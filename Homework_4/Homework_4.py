num_1 = "0c60c80f961f0e71f3a9b524af6012062fe037a6"
num_2 = "e60cc942513261fd3eb76c0e617d53f6f73ebef1"
c = ""
for i, hex_digit in enumerate(num_1):
    a = int(hex_digit, 16)
    b = int(num_2[i], 16)
    print(a)
    print(b)
    c = c + str(hex(a ^ b))[2]
print(c)

