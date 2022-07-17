data = [(20000, 0.165, 0.565), (50000, 0.1818, 0.6193), (100000, 0.1992, 0.677), (150000, 0.2115, 0.7189), (200000, 0.2162, 0.7527), (250000, 0.2272, 0.7796), (300000, 0.2358, 0.8004), (350000, 0.2394, 0.816), (400000, 0.2442, 0.8293), (450000, 0.2488, 0.8405), (500000, 0.2523, 0.8488), (550000, 0.2574, 0.8574), (600000, 0.259, 0.8643), (650000, 0.2626, 0.8706), (700000, 0.2657, 0.8763), (750000, 0.2698, 0.8811), (800000, 0.2744, 0.8867), (850000, 0.2803, 0.8899), (900000, 0.2823, 0.8935), (950000, 0.2855, 0.8974), (1000000, 0.2894, 0.8992), (1050000, 0.2923, 0.9018), (1100000, 0.2945, 0.9042), (1150000, 0.2971, 0.9065), (1200000, 0.2991, 0.9094), (1250000, 0.3008, 0.9116), (1300000, 0.3022, 0.9127), (1350000, 0.3031, 0.9148), (1400000, 0.3038, 0.9168), (1450000, 0.3053, 0.9185), (1500000, 0.3075, 0.9203), (1550000, 0.3092, 0.9221), (1600000, 0.3102, 0.9235), (1650000, 0.3115, 0.9255), (1700000, 0.3125, 0.9268), (1750000, 0.3132, 0.9286), (1800000, 0.3131, 0.9297), (1850000, 0.3152, 0.931), (1900000, 0.3172, 0.9324), (1950000, 0.3183, 0.9336), (2000000, 0.319, 0.9338), (2050000, 0.3192, 0.9349), (2100000, 0.32, 0.9359), (2150000, 0.321, 0.9369), (2200000, 0.3219, 0.9378), (2250000, 0.3225, 0.9387), (2300000, 0.3234, 0.9395), (2350000, 0.3242, 0.9406), (2400000, 0.3251, 0.9415), (2450000, 0.3259, 0.9425), (2500000, 0.3256, 0.9433), (2550000, 0.3265, 0.9442), (2600000, 0.3273, 0.9451), (2650000, 0.3282, 0.9454), (2700000, 0.3286, 0.9464), (2750000, 0.3263, 0.9449), (2800000, 0.3263, 0.9453), (2850000, 0.3252, 0.945), (2900000, 0.3265, 0.9456), (2950000, 0.3268, 0.946), (3000000, 0.3256, 0.9439), (3050000, 0.3259, 0.9445), (3100000, 0.3264, 0.9451), (3150000, 0.3265, 0.9461), (3200000, 0.3294, 0.9475), (3250000, 0.3297, 0.9477)]
base_data = [(20000, 0.1607, 0.5482), (50000, 0.1717, 0.5974), (100000, 0.1847, 0.6502), (150000, 0.1942, 0.6881), (200000, 0.1985, 0.7172), (250000, 0.2029, 0.741), (300000, 0.2047, 0.7582), (350000, 0.2077, 0.7741), (400000, 0.2112, 0.7857), (450000, 0.2107, 0.7929), (500000, 0.2126, 0.8005), (550000, 0.2153, 0.8052), (600000, 0.2178, 0.809), (650000, 0.2207, 0.8152), (700000, 0.2211, 0.8221), (750000, 0.2229, 0.8258), (800000, 0.2236, 0.8304), (850000, 0.2196, 0.8294), (900000, 0.2208, 0.8332), (950000, 0.2219, 0.8361), (1000000, 0.2222, 0.8394), (1050000, 0.2224, 0.8424), (1100000, 0.2229, 0.8444), (1150000, 0.2237, 0.8469), (1200000, 0.2234, 0.8487), (1250000, 0.223, 0.8511), (1300000, 0.2227, 0.8529), (1350000, 0.2232, 0.8546), (1400000, 0.223, 0.8565), (1450000, 0.223, 0.8583), (1500000, 0.223, 0.8597), (1550000, 0.2226, 0.8615), (1600000, 0.2228, 0.8629), (1650000, 0.2221, 0.8649), (1700000, 0.2213, 0.8664), (1750000, 0.2213, 0.868), (1800000, 0.2215, 0.8696), (1850000, 0.221, 0.871), (1900000, 0.2211, 0.8726), (1950000, 0.2198, 0.8716), (2000000, 0.22, 0.8723), (2050000, 0.221, 0.8734), (2100000, 0.2204, 0.8741)]
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

# ax1.plot([item[0] for item in data], [item[1] for item in data], label='Unknown')
# ax1.plot([item[0] for item in data], [item[2] for item in data], label='Known')
# ax1.set_ylabel('+ Self-Training')
# ax1.legend()
#
# ax2.plot([item[0] for item in base_data], [item[1] for item in base_data], label='Unknown')
# ax2.plot([item[0] for item in base_data], [item[2] for item in base_data], label='Known')
#
# ax2.set_ylabel('Online Concept Discovery')
# ax2.set_xlabel('iterations')


ax1.plot([item[0] for item in data], [item[1] for item in data], label='+ Self-Training')
ax1.plot([item[0] for item in base_data], [item[1] for item in base_data], label='Online Concept Discovery')

ax1.set_ylabel('Unknown Concepts')
ax1.set_xlabel('iterations')
ax1.legend()

ax2.plot([item[0] for item in data], [item[2] for item in data], label='+ Self-Training')
ax2.plot([item[0] for item in base_data], [item[2] for item in base_data], label='Online Concept Discovery')

ax2.legend()
ax2.set_ylabel('Known Concepts')
ax2.set_xlabel('iterations')

plt.savefig('converge_self_training.pdf')
plt.show()