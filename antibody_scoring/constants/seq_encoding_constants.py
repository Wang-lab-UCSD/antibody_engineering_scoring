"""Constants for use in encoding sequences."""

DESAUTELS_VARIABLE_POSITIONS = [30, 31, 32, 46, 49, 50, 51, 53, 54, 56,
        57, 58, 59, 60, 61, 98, 99, 100, 101, 102, 103, 270, 272, 273,
        274, 334, 335, 336, 337, 339, 340]

IL6_ANTIGEN_VARIABLE_POSITIONS = [41, 44, 47, 50, 53, 56, 59, 62, 68, 71, 77, 80, 86, 89, 92, 98,
        101, 104, 107, 116, 119, 125, 128, 134, 137, 143, 152, 161, 164, 167]


aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

"""Physicochemical properties (from Grantham et al.)"""
COMPOSITION = {'A': -0.887315380986987, 'R': 0.041436628097621, 'N': 1.0130541145246, 'D': 1.08449657676187, 'C': 3.04202004206328, 'Q': 0.384360446836553, 'E': 0.42722592417892, 'G': 0.17003306012472, 'H': -0.0585828190345676, 'I': -0.887315380986987, 'L': -0.887315380986987, 'K': -0.415795130220955, 'M': -0.887315380986987, 'F': -0.887315380986987, 'P': -0.330064175536222, 'S': 1.14165054655169, 'T': 0.127167582782354, 'W': -0.701564979170065, 'Y': -0.601545532037877, 'V': -0.887315380986987, '-':0.0}
VOLUME = {'A': -1.23448897385975, 'R': 0.930668490901342, 'N': -0.652457397311072, 'D': -0.699019923434967, 'C': -0.67573866037302, 'Q': 0.0226992314853985, 'E': -0.0238632946384961, 'G': -1.88636433959428, 'H': 0.278793125166818, 'I': 0.628012071096028, 'L': 0.628012071096028, 'K': 0.814262175591606, 'M': 0.488324492724344, 'F': 1.11691859539692, 'P': -1.19956707926683, 'S': -1.21120771079781, 'T': -0.536051082001336, 'W': 2.00160659175092, 'Y': 1.21004364764471, 'V': -0.0005820315765488, '-':0.0}
POLARITY = {'A': -0.0836274309924444, 'R': 0.808398499593631, 'N': 1.21724371777891, 'D': 1.73759217728746, 'C': -1.04998885579403, 'Q': 0.808398499593631, 'E': 1.47741794753319, 'G': 0.250882292977334, 'H': 0.771230752485878, 'I': -1.16149209711728, 'L': -1.27299533844054, 'K': 1.10574047645566, 'M': -0.975653361578519, 'F': -1.16149209711728, 'P': -0.120795178100197, 'S': 0.32521778719284, 'T': 0.102211304546321, 'W': -1.08715660290178, 'Y': -0.789814626039753, 'V': -0.901317867363013, '-':0.0}
