#prompt (not used)
prompts = []
y_test = []
num_shots = params['num_shots']
help_dict = {
'0': 'zero',
'1': 'one',
'2': 'two',
'3': 'three',
'4': 'four',
'5': 'five',
'6': 'six',
'7': 'seven'
}
for i in range(len(D_te.y)):
    test_str  = "Eight 8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Class: {}".format(''.join(help_dict[ele] for ele in str(D_full.y.numpy()[j])))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Class: "
    prompts.append(test_str)
    y_test.append(D_te.y.numpy()[i])
    # print("\n\ntest_str: ", test_str)
params['label_dict'] = {0: ['zero'], 1: ['one'], 2: ['two'], 3: ['three'], 4: ['four'], 5: ['five'], 6: ['six'], 7: ['seven']}

prompts = []
y_test = []
num_shots = params['num_shots']
help_dict = {
'0': 'zero',
'1': 'one',
'2': 'two',
'3': 'three',
'4': 'four',
'5': 'five',
'6': 'six',
'7': 'seven'
}
for i in range(len(D_full.y)):
    test_str  = "Eight 8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Class: {}".format(''.join(help_dict[ele] for ele in str(D_full.y.numpy()[j])))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Class: "
    prompts.append(test_str)
    y_test.append(D_full.y.numpy()[i])
params['label_dict'] = {0: ['zero'], 1: ['one'], 2: ['two'], 3: ['three'], 4: ['four'], 5: ['five'], 6: ['six'], 7: ['seven']}


##prompt 1
prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_te.y)):
    test_str  = "8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s real part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's real part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and imaginery part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Signal: "
    prompts.append(test_str)
    y_test.append(D_te.y.numpy()[i])
    # print("\n\ntest_str: ", test_str)
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_full.y)):
    test_str  = "8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s real part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's real part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Signal: "
    prompts.append(test_str)
    y_test.append(D_full.y.numpy()[i])
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}


#prompt 2
prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_te.y)):
    test_str  = "8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s real part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Constellation: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's real part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and imaginery part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Constellation: "
    prompts.append(test_str)
    y_test.append(D_te.y.numpy()[i])
    # print("\n\ntest_str: ", test_str)
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_full.y)):
    test_str  = "8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s real part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Constellation: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's real part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Constellation: "
    prompts.append(test_str)
    y_test.append(D_full.y.numpy()[i])
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}



#prompt3
prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_te.y)):
    test_str  = "8APSK signals are as follows. Classifiy the signals based on the true set of classes [0, 1, 2, 3, 4, 5, 6, 7]."
    for j in range(num_shots):
        test_str += "\nSignal#{}'s real part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's real part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and imaginery part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Signal: "
    prompts.append(test_str)
    y_test.append(D_te.y.numpy()[i])
    # print("\n\ntest_str: ", test_str)
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_full.y)):
    test_str  = "8APSK signals are as follows. Classifiy the signals based on the true set of classes [0, 1, 2, 3, 4, 5, 6, 7]."
    for j in range(num_shots):
        test_str += "\nSignal#{}'s real part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's real part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Signal: "
    prompts.append(test_str)
    y_test.append(D_full.y.numpy()[i])
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}



#prompt4
prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_te.y)):
    test_str  = "Based on the 8APSK signals shown below, predict the Test Signal's output class from the set of classes [0, 1, 2, 3, 4, 5, 6, 7]:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s real part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's real part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and imaginery part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Signal: "
    prompts.append(test_str)
    y_test.append(D_te.y.numpy()[i])
    # print("\n\ntest_str: ", test_str)
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_full.y)):
    test_str  = "Based on the 8APSK signals shown below, predict the Test Signal's output class from the set of classes [0, 1, 2, 3, 4, 5, 6, 7]:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s real part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's real part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Signal: "
    prompts.append(test_str)
    y_test.append(D_full.y.numpy()[i])
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}



#prompt5
prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_te.y)):
    test_str  = "8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s real part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Constellation Signal: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's real part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and imaginery part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Constellation Signal: "
    prompts.append(test_str)
    y_test.append(D_te.y.numpy()[i])
    # print("\n\ntest_str: ", test_str)
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_full.y)):
    test_str  = "8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s real part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Constellation Signal: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's real part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Constellation Signal: "
    prompts.append(test_str)
    y_test.append(D_full.y.numpy()[i])
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}



#prompt6
prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_te.y)):
    test_str  = "8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Signal: "
    prompts.append(test_str)
    y_test.append(D_te.y.numpy()[i])
    # print("\n\ntest_str: ", test_str)
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_full.y)):
    test_str  = "8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Signal: "
    prompts.append(test_str)
    y_test.append(D_full.y.numpy()[i])
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}



#prompt7
prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_te.y)):
    test_str  = "Based on the 8APSK signals shown below, predict the Test Signal's output class from the set of classes [0, 1, 2, 3, 4, 5, 6, 7]:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Signal: "
    prompts.append(test_str)
    y_test.append(D_te.y.numpy()[i])
    # print("\n\ntest_str: ", test_str)
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_full.y)):
    test_str  = "Based on the 8APSK signals shown below, predict the Test Signal's output class from the set of classes [0, 1, 2, 3, 4, 5, 6, 7]:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Signal: "
    prompts.append(test_str)
    y_test.append(D_full.y.numpy()[i])
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}



#prompt8
prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_te.y)):
    test_str  = "Eight 8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Signal: "
    prompts.append(test_str)
    y_test.append(D_te.y.numpy()[i])
    # print("\n\ntest_str: ", test_str)
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_full.y)):
    test_str  = "Eight 8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Signal: "
    prompts.append(test_str)
    y_test.append(D_full.y.numpy()[i])
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}



#prompt9
prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_te.y)):
    test_str  = "Eight 8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Class: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Class: "
    prompts.append(test_str)
    y_test.append(D_te.y.numpy()[i])
    # print("\n\ntest_str: ", test_str)
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_full.y)):
    test_str  = "Eight 8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Class: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Class: "
    prompts.append(test_str)
    y_test.append(D_full.y.numpy()[i])
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}



#prompt10
prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_te.y)):
    test_str  = "8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Class: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Class: "
    prompts.append(test_str)
    y_test.append(D_te.y.numpy()[i])
    # print("\n\ntest_str: ", test_str)
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_full.y)):
    test_str  = "8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Class: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Class: "
    prompts.append(test_str)
    y_test.append(D_full.y.numpy()[i])
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}



#prompt11
prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_te.y)):
    test_str  = "8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Constellation: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Constellation: "
    prompts.append(test_str)
    y_test.append(D_te.y.numpy()[i])
    # print("\n\ntest_str: ", test_str)
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_full.y)):
    test_str  = "8APSK signals are as follows:"
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Constellation: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Constellation: "
    prompts.append(test_str)
    y_test.append(D_full.y.numpy()[i])
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}



#prompt12
prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_te.y)):
    test_str  = "8APSK signals are as follows. Classifiy the signals based on the true set of classes [0, 1, 2, 3, 4, 5, 6, 7]."
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Class: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Class: "
    prompts.append(test_str)
    y_test.append(D_te.y.numpy()[i])
    # print("\n\ntest_str: ", test_str)
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_full.y)):
    test_str  = "8APSK signals are as follows. Classifiy the signals based on the true set of classes [0, 1, 2, 3, 4, 5, 6, 7]."
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Class: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Class: "
    prompts.append(test_str)
    y_test.append(D_full.y.numpy()[i])
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}



#prompt13
prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_te.y)):
    test_str  = "Eight 8APSK signals are as follows. Classifiy the signals based on the true set of classes [0, 1, 2, 3, 4, 5, 6, 7]."
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Class: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Class: "
    prompts.append(test_str)
    y_test.append(D_te.y.numpy()[i])
    # print("\n\ntest_str: ", test_str)
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

prompts = []
y_test = []
num_shots = params['num_shots']
for i in range(len(D_full.y)):
    test_str  = "Eight 8APSK signals are as follows. Classifiy the signals based on the true set of classes [0, 1, 2, 3, 4, 5, 6, 7]."
    for j in range(num_shots):
        test_str += "\nSignal#{}'s in-phase part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Class: {}".format(str(D_full.y.numpy()[j]))

    test_str += "\nTest Signal's in-phase part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and quadrature part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Class: "
    prompts.append(test_str)
    y_test.append(D_full.y.numpy()[i])
params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}





