import torch
from torch import nn
import itertools

from torch.autograd.variable import Variable


class GraphNet(nn.Module):
    """
    Model definition taken from https://github.com/vlimant/NNLO/blob/master/examples/example_jedi_torch.py
    """

    def __init__(
        self,
        n_constituents,
        n_targets,
        params,
        hidden,
        De,
        Do,
        fr_activation=0,
        fo_activation=0,
        fc_activation=0,
        optimizer=0,
        verbose=False,
        device="cpu",
        sumO=True,
    ):
        super(GraphNet, self).__init__()
        self.device = device
        self.hidden = hidden
        self.P = len(params)
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.n_targets = n_targets
        self.fr_activation = fr_activation
        self.fo_activation = fo_activation
        self.fc_activation = fc_activation
        self.optimizer = optimizer
        self.verbose = verbose
        self.assign_matrices()

        self.sum_O = sumO
        self.Ra = torch.ones(self.Dr, self.Nr)
        if self.device == "cuda":
            self.fr1 = nn.Linear(2 * self.P + self.Dr, hidden).cuda()
            self.fr2 = nn.Linear(hidden, int(hidden / 2)).cuda()
            self.fr3 = nn.Linear(int(hidden / 2), self.De).cuda()
            self.fo1 = nn.Linear(self.P + self.Dx + self.De, hidden).cuda()
            self.fo2 = nn.Linear(hidden, int(hidden / 2)).cuda()
            self.fo3 = nn.Linear(int(hidden / 2), self.Do).cuda()
            if self.sum_O:
                self.fc1 = nn.Linear(self.Do * 1, hidden).cuda()
            else:
                self.fc1 = nn.Linear(self.Do * self.N, hidden).cuda()
            self.fc2 = nn.Linear(hidden, int(hidden / 2)).cuda()
            self.fc3 = nn.Linear(int(hidden / 2), self.n_targets).cuda()
        else:
            self.fr1 = nn.Linear(2 * self.P + self.Dr, hidden)
            self.fr2 = nn.Linear(hidden, int(hidden / 2))
            self.fr3 = nn.Linear(int(hidden / 2), self.De)
            self.fo1 = nn.Linear(self.P + self.Dx + self.De, hidden)
            self.fo2 = nn.Linear(hidden, int(hidden / 2))
            self.fo3 = nn.Linear(int(hidden / 2), self.Do)
            if self.sum_O:
                self.fc1 = nn.Linear(self.Do * 1, hidden)
            else:
                self.fc1 = nn.Linear(self.Do * self.N, hidden)
            self.fc2 = nn.Linear(hidden, int(hidden / 2))
            self.fc3 = nn.Linear(int(hidden / 2), self.n_targets)
        print("GraphNet created")

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [
            i for i in itertools.product(range(self.N), range(self.N)) if i[0] != i[1]
        ]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        if self.device == "cuda":
            self.Rr = Variable(self.Rr).cuda()
            self.Rs = Variable(self.Rs).cuda()
        else:
            self.Rr = Variable(self.Rr)
            self.Rs = Variable(self.Rs)

    def forward(self, x):
        Orr = self.tmul(x.float(), self.Rr)
        Ors = self.tmul(x.float(), self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        if self.fr_activation == 2:
            B = nn.functional.selu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.selu(self.fr2(B))
            E = nn.functional.selu(self.fr3(B).view(-1, self.Nr, self.De))
        elif self.fr_activation == 1:
            B = nn.functional.elu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.elu(self.fr2(B))
            E = nn.functional.elu(self.fr3(B).view(-1, self.Nr, self.De))
        else:
            B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.relu(self.fr2(B))
            E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        C = torch.cat([x.float(), Ebar], 1)
        del Ebar
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        if self.fo_activation == 2:
            C = nn.functional.selu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.selu(self.fo2(C))
            O = nn.functional.selu(self.fo3(C).view(-1, self.N, self.Do))
        elif self.fo_activation == 1:
            C = nn.functional.elu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.elu(self.fo2(C))
            O = nn.functional.elu(self.fo3(C).view(-1, self.N, self.Do))
        else:
            C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.relu(self.fo2(C))
            O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C
        ## sum over the O matrix
        if self.sum_O:
            O = torch.sum(O, dim=1)
        ### Classification MLP ###
        if self.fc_activation == 2:
            if self.sum_O:
                N = nn.functional.selu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.selu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.selu(self.fc2(N))
        elif self.fc_activation == 1:
            if self.sum_O:
                N = nn.functional.elu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.elu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.elu(self.fc2(N))
        else:
            if self.sum_O:
                N = nn.functional.relu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.relu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.relu(self.fc2(N))
        del O
        # N = nn.functional.relu(self.fc3(N))
        N = self.fc3(N)
        return N

    def tmul(self, x, y):  # Takes (I * J * K)(K * L) -> I * J * L
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])


def get_model(sumO, device):

    nParticles = 150
    labels = ["j_g", "j_q", "j_w", "j_z", "j_t"]
    params = [
        "j1_px",
        "j1_py",
        "j1_pz",
        "j1_e",
        "j1_erel",
        "j1_pt",
        "j1_ptrel",
        "j1_eta",
        "j1_etarel",
        "j1_etarot",
        "j1_phi",
        "j1_phirel",
        "j1_phirot",
        "j1_deltaR",
        "j1_costheta",
        "j1_costhetarel",
    ]

    load = True
    args = {}
    if load:
        if sumO:
            x = [50, 14, 10, 2, 2, 2, 0]
        else:
            x = [10, 4, 14, 2, 2, 2, 0]
        ## load the best model for 150 particles
        args.setdefault("hidden", x[0])
        args.setdefault("De", x[1])
        args.setdefault("Do", x[2])
        args.setdefault("fr_act", x[3])
        args.setdefault("fo_act", x[4])
        args.setdefault("fc_act", x[5])

    mymodel = GraphNet(
        nParticles,
        len(labels),
        params,
        hidden=args.get("hidden", 10),
        De=args.get("De", 10),
        Do=args.get("Do", 10),
        fr_activation=args.get("fr_act", 0),
        fo_activation=args.get("fo_act", 0),
        fc_activation=args.get("fc_act", 0),
        optimizer=0,  # disabled
        verbose=True,
        device=device,
        sumO=sumO,
    )
    return mymodel


class GraphNetOld(nn.Module):
    """
    Model definition taken from https://github.com/jmduarte/JEDInet-code/blob/master/python/gnn_top.py
    """

    def __init__(
        self,
        n_constituents,
        n_targets,
        params,
        hidden,
        De,
        Do,
        fr_activation=0,
        fo_activation=0,
        fc_activation=0,
        optimizer=0,
        verbose=False,
        device="cpu",
        sum_O=True,
    ):
        super(GraphNetOld, self).__init__()
        self.device = device
        self.hidden = hidden
        self.P = len(params)
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.n_targets = n_targets
        self.fr_activation = fr_activation
        self.fo_activation = fo_activation
        self.fc_activation = fc_activation
        self.optimizer = optimizer
        self.verbose = verbose
        self.assign_matrices()

        self.sum_O = sum_O
        self.Ra = torch.ones(self.Dr, self.Nr)
        if self.device == "cpu":
            self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden)
            self.fr2 = nn.Linear(self.hidden, int(self.hidden / 2))
            self.fr3 = nn.Linear(int(self.hidden / 2), self.De)
            self.fo1 = nn.Linear(self.P + self.Dx + self.De, self.hidden)
            self.fo2 = nn.Linear(self.hidden, int(self.hidden / 2))
            self.fo3 = nn.Linear(int(self.hidden / 2), self.Do)
            if self.sum_O:
                self.fc1 = nn.Linear(self.Do * 1, self.hidden)
            else:
                self.fc1 = nn.Linear(self.Do * self.N, self.hidden)
            self.fc2 = nn.Linear(self.hidden, int(self.hidden / 2))
            self.fc3 = nn.Linear(int(self.hidden / 2), self.n_targets)
        elif self.device == "cuda":
            self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden).cuda()
            self.fr2 = nn.Linear(self.hidden, int(self.hidden / 2)).cuda()
            self.fr3 = nn.Linear(int(self.hidden / 2), self.De).cuda()
            self.fo1 = nn.Linear(self.P + self.Dx + self.De, self.hidden).cuda()
            self.fo2 = nn.Linear(self.hidden, int(self.hidden / 2)).cuda()
            self.fo3 = nn.Linear(int(self.hidden / 2), self.Do).cuda()
            if self.sum_O:
                self.fc1 = nn.Linear(self.Do * 1, self.hidden).cuda()
            else:
                self.fc1 = nn.Linear(self.Do * self.N, self.hidden).cuda()
            self.fc2 = nn.Linear(self.hidden, int(self.hidden / 2)).cuda()
            self.fc3 = nn.Linear(int(self.hidden / 2), self.n_targets).cuda()
        else:
            raise ValueError("device has to be either 'cuda' or 'cpu'!")

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [
            i for i in itertools.product(range(self.N), range(self.N)) if i[0] != i[1]
        ]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        if self.device == "cpu":
            self.Rr = self.Rr
            self.Rs = self.Rs
        elif self.device == "cuda":
            self.Rr = self.Rr.cuda()
            self.Rs = self.Rs.cuda()

    def forward(self, x):
        Orr = self.tmul(x.float(), self.Rr)
        Ors = self.tmul(x.float(), self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        if self.fr_activation == 2:
            B = nn.functional.selu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.selu(self.fr2(B))
            E = nn.functional.selu(self.fr3(B).view(-1, self.Nr, self.De))
        elif self.fr_activation == 1:
            B = nn.functional.elu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.elu(self.fr2(B))
            E = nn.functional.elu(self.fr3(B).view(-1, self.Nr, self.De))
        else:
            B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.relu(self.fr2(B))
            E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        C = torch.cat([x.float(), Ebar], 1)
        del Ebar
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        if self.fo_activation == 2:
            C = nn.functional.selu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.selu(self.fo2(C))
            O = nn.functional.selu(self.fo3(C).view(-1, self.N, self.Do))
        elif self.fo_activation == 1:
            C = nn.functional.elu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.elu(self.fo2(C))
            O = nn.functional.elu(self.fo3(C).view(-1, self.N, self.Do))
        else:
            C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.relu(self.fo2(C))
            O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C
        ## sum over the O matrix
        if self.sum_O:
            O = torch.sum(O, dim=1)
        ### Classification MLP ###
        if self.fc_activation == 2:
            if self.sum_O:
                N = nn.functional.selu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.selu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.selu(self.fc2(N))
        elif self.fc_activation == 1:
            if self.sum_O:
                N = nn.functional.elu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.elu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.elu(self.fc2(N))
        else:
            if self.sum_O:
                N = nn.functional.relu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.relu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.relu(self.fc2(N))
        del O
        # N = nn.functional.relu(self.fc3(N))
        N = self.fc3(N)
        return N

    def tmul(self, x, y):  # Takes (I * J * K)(K * L) -> I * J * L
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])


def get_model_old(sumO, device):

    nParticles = 150
    labels = ["j_g", "j_q", "j_w", "j_z", "j_t"]
    params = [
        "j1_px",
        "j1_py",
        "j1_pz",
        "j1_e",
        "j1_erel",
        "j1_pt",
        "j1_ptrel",
        "j1_eta",
        "j1_etarel",
        "j1_etarot",
        "j1_phi",
        "j1_phirel",
        "j1_phirot",
        "j1_deltaR",
        "j1_costheta",
        "j1_costhetarel",
    ]

    load = True
    args = {}
    if load:
        if sumO:
            x = [50, 14, 10, 2, 2, 2, 0]
        else:
            x = [10, 4, 14, 2, 2, 2, 0]
        ## load the best model for 150 particles
        args.setdefault("hidden", x[0])
        args.setdefault("De", x[1])
        args.setdefault("Do", x[2])
        args.setdefault("fr_act", x[3])
        args.setdefault("fo_act", x[4])
        args.setdefault("fc_act", x[5])

    mymodel = GraphNetOld(
        nParticles,
        len(labels),
        params,
        hidden=args.get("hidden", 10),
        De=args.get("De", 10),
        Do=args.get("Do", 10),
        fr_activation=args.get("fr_act", 0),
        fo_activation=args.get("fo_act", 0),
        fc_activation=args.get("fc_act", 0),
        optimizer=0,  # disabled
        verbose=True,
        device=device,
        sum_O=sumO,
    )
    return mymodel


def get_model_from_config(config):
    if config["model_name"] == "GraphNetOld":
        return get_model_old(config["sumO"], config["device"])
    elif config["model_name"] == "GraphNet":
        return get_model(config["sumO"], config["device"])
