import numpy as np
from sympy import Matrix, zeros, Symbol


# Main object
class Robot:

    def __init__(self, name: str = ""):
        """Object constructor

        Args:
          name (str, optional): robot's name (if any)
        """
        self.name = name


# Inherited
class Serial(Robot):
    """Serial Robot

    Args:
      Robot (obj): inheritance
    """

    def __init__(
        self,
        jointsPositions: np.ndarray,
        jointsVelocities: np.ndarray,
        jointsAccelerations: np.ndarray,
        linksLengths: list,
        COMs: list,
        mass: list,
        inertia: list,
        xi=None,
        xid=None,
        name: str = "",
        user_DH: Matrix = None,
        user_DH_COM: Matrix = None,
        symbolic: bool = False,
    ):
        """Object constructor (API unchanged)"""
        super().__init__(name=name)

        # --- helpers to normalize shape depending on mode ---
        def _col_np(a):
            a = np.asarray(a, dtype=float)
            if a.ndim == 1:
                a = a.reshape((-1, 1))
            return a

        def _col_sp(a):
            if isinstance(a, Matrix):
                A = a
            else:
                A = Matrix(a)
            # ensure column vector
            if A.shape[1] != 1 and A.shape[0] == 1:
                A = A.T
            return A

        self.symbolic = symbolic

        # preserve inputs with correct backend types
        if self.symbolic:
            self.jointsPositions = _col_sp(jointsPositions)
            self.jointsVelocities = _col_sp(jointsVelocities)
            self.jointsAccelerations = _col_sp(jointsAccelerations)
            # keep kinematic params as python/sympy lists (can be Symbols)
            self.linksLengths = list(linksLengths)
            self.COMs = list(COMs)
            self.mass = list(mass)
            self.inertia = list(inertia)  # list of SymPy 3x3
        else:
            self.jointsPositions = _col_np(jointsPositions)
            self.jointsVelocities = _col_np(jointsVelocities)
            self.jointsAccelerations = _col_np(jointsAccelerations)
            self.linksLengths = list(linksLengths)
            self.COMs = list(COMs)
            self.mass = list(mass)
            self.inertia = [np.asarray(I, dtype=float) for I in inertia]

        self.xi = xi if xi is not None else []
        self.xid = xid if xid is not None else []
        self.user_DH = user_DH
        self.user_DH_COM = user_DH_COM

        # number of joints
        self.n = int(self.jointsPositions.shape[0])

        # numeric quaternion-form inertia (only for numeric mode)
        if not self.symbolic:
            self.quaternionInertia = [
                np.vstack(
                    [
                        np.hstack((np.eye(1), np.zeros((1, 3)))),
                        np.hstack((np.zeros((3, 1)), I)),
                    ]
                )
                for I in self.inertia
            ]
        else:
            self.quaternionInertia = None  # use symbolicQuaternionInertia instead

        # symbolic variables (always created; used when symbolic=True)
        self.qSymbolic = Matrix([Symbol(f"q{i+1}") for i in range(self.n)])
        self.qdSymbolic = Matrix([Symbol(f"qd{i+1}") for i in range(self.n)])
        self.qddSymbolic = Matrix([Symbol(f"qdd{i+1}") for i in range(self.n)])
        self.symbolicLinks = Matrix([Symbol(f"L{i+1}") for i in range(self.n)])
        self.symbolicCOMs = Matrix([Symbol(f"Lcom{i+1}") for i in range(self.n)])
        self.symbolicMass = Matrix([Symbol(f"m{i+1}") for i in range(self.n)])
        self.symbolicInertia = [
            Matrix(
                [
                    [Symbol(f"Ixx{i+1}"), -Symbol(f"Ixy{i+1}"), -Symbol(f"Ixz{i+1}")],
                    [-Symbol(f"Ixy{i+1}"), Symbol(f"Iyy{i+1}"), -Symbol(f"Iyz{i+1}")],
                    [-Symbol(f"Ixz{i+1}"), -Symbol(f"Iyz{i+1}"), Symbol(f"Izz{i+1}")],
                ]
            )
            for i in range(self.n)
        ]
        self.symbolicQuaternionInertia = [
            Matrix([[1, 0, 0, 0]]).row_insert(1, zeros(3, 1).col_insert(1, SI))
            for SI in self.symbolicInertia
        ]

        # lightweight caches / maps to accelerate repeated queries
        self._cache_q = None
        self._cache_links = None
        self._cache_coms = None
        self._sym_q_posmap = {}  # {Symbol('qk'): (row, col)}
        self._sym_lcom_posmap = {}  # {Symbol('Lcomk'): (row, col)}

        # build both numeric and symbolic DH tables
        self.denavitHartenberg(symbolic=self.symbolic)
        self.denavitHartenbergCOM(symbolic=self.symbolic)

    def denavitHartenberg(self, symbolic=False):
        """Denavit–Hartenberg parameters for i-th frame."""
        rows = self.n + 1

        if symbolic:
            if self.user_DH is not None:
                self.symbolicDHParameters = Matrix(self.user_DH)
            else:
                theta = [0] + list(self.qSymbolic)
                d = [0] * rows
                a = [0] + list(self.symbolicLinks)
                alpha = [0] * rows
                self.symbolicDHParameters = Matrix(
                    [[theta[i], d[i], a[i], alpha[i]] for i in range(rows)]
                )
            # build a lookup map once for quick whereIsTheJoint()
            self._sym_q_posmap = {}
            for r in range(self.symbolicDHParameters.rows):
                for c in range(self.symbolicDHParameters.cols):
                    v = self.symbolicDHParameters[r, c]
                    if isinstance(v, Symbol) and v.name.startswith("q"):
                        self._sym_q_posmap[v] = (r, c)
        else:
            if self.user_DH is not None:
                self.dhParameters = np.array(self.user_DH, dtype=float)
            else:
                qflat = self.jointsPositions[:, 0]
                if (
                    getattr(self, "dhParameters", None) is not None
                    and self._cache_q is not None
                    and self._cache_links is not None
                    and np.array_equal(self._cache_q, qflat)
                    and self._cache_links == tuple(self.linksLengths)
                ):
                    return  # cache valid
                theta = np.concatenate(([0.0], qflat))
                d = np.zeros(rows, float)
                a = np.concatenate(([0.0], np.asarray(self.linksLengths, float)))
                alpha = np.zeros(rows, float)
                self.dhParameters = np.column_stack((theta, d, a, alpha))
                # update cache keys
                self._cache_q = theta[1:].copy()
                self._cache_links = tuple(self.linksLengths)

    def denavitHartenbergCOM(self, symbolic=False):
        """Denavit–Hartenberg parameters for COM frames."""
        rows = self.n + 1

        if symbolic:
            if self.user_DH_COM is not None:
                self.symbolicDHParametersCOM = Matrix(self.user_DH_COM)
            else:
                theta = [0] + list(self.qSymbolic)
                d = [0] * rows
                a = [0] + list(self.symbolicCOMs)
                alpha = [0] * rows
                self.symbolicDHParametersCOM = Matrix(
                    [[theta[i], d[i], a[i], alpha[i]] for i in range(rows)]
                )
            # quick lookup map for whereIsTheCOM()
            self._sym_lcom_posmap = {}
            for r in range(self.symbolicDHParametersCOM.rows):
                for c in range(self.symbolicDHParametersCOM.cols):
                    v = self.symbolicDHParametersCOM[r, c]
                    if isinstance(v, Symbol) and v.name.startswith("Lcom"):
                        self._sym_lcom_posmap[v] = (r, c)
        else:
            if self.user_DH_COM is not None:
                self.dhParametersCOM = np.array(self.user_DH_COM, dtype=float)
            else:
                qflat = self.jointsPositions[:, 0]
                if (
                    getattr(self, "dhParametersCOM", None) is not None
                    and self._cache_q is not None
                    and self._cache_coms is not None
                    and np.array_equal(self._cache_q, qflat)
                    and self._cache_coms == tuple(self.COMs)
                ):
                    return  # cache valid
                theta = np.concatenate(([0.0], qflat))
                d = np.zeros(rows, float)
                a = np.concatenate(([0.0], np.asarray(self.COMs, float)))
                alpha = np.zeros(rows, float)
                self.dhParametersCOM = np.column_stack((theta, d, a, alpha))
                # update cache keys (reuse q cache)
                self._cache_q = theta[1:].copy()
                self._cache_coms = tuple(self.COMs)

    def whereIsTheJoint(self, joint: int):
        """
        Locate joint number `joint` in the DH table.
        - symbolic=True: search for Symbol('q{joint}') in symbolicDHParameters
        - symbolic=False: θ lives in row=joint, col=0 (joint indices are 1..n)
        """
        if self.symbolic:
            target = Symbol(f"q{joint}")
            # O(1) if map available, fallback to scan if user modified the DH externally
            if target in self._sym_q_posmap:
                return self._sym_q_posmap[target]
            for i, row in enumerate(self.symbolicDHParameters.tolist()):
                if target in row:
                    return i, row.index(target)
            raise ValueError(f"q{joint} not found")
        else:
            if 1 <= joint <= self.n:
                return joint, 0
            else:
                raise IndexError(f"joint index {joint} out of range (1..{self.n})")

    def whereIsTheCOM(self, COM: int):
        """
        Locate COM number `COM` in the COM‐DH table.
        - symbolic=True: search for Symbol('Lcom{COM}') in symbolicDHParametersCOM
        - symbolic=False: COM i is at DH‐COM row i, a-column = 2 (COM indices are 1..n)
        """
        if self.symbolic:
            target = Symbol(f"Lcom{COM}")
            if target in self._sym_lcom_posmap:
                return self._sym_lcom_posmap[target]
            for i, row in enumerate(self.symbolicDHParametersCOM.tolist()):
                if target in row:
                    return i, row.index(target)
            raise ValueError(f"Lcom{COM} not found")
        else:
            if 1 <= COM <= self.n:
                return COM, 2
            else:
                raise IndexError(f"COM index {COM} out of range (1..{self.n})")


if __name__ == "__main__":
    # your original testing block, corrected:
    from sympy import symbols, Matrix, pi, pprint

    # define q-symbols
    q1, q2 = symbols("q1 q2")
    # define L-symbols for your user_DH
    L1, L2 = symbols("L1 L2")

    user_DH_sym = Matrix([[0, 0, 0, 0], [q1, 0, L1, 0], [q2, 0, L2, 0]])
    user_DH_num = np.array([[0, 0, 0, 0], [pi / 4, 0, 1, 0], [pi / 6, 0, 1, 0]])

    jointsPositions = np.array([[pi / 4], [pi / 6]])
    jointsVelocities = np.array([[0.1], [0.2]])
    jointsAccelerations = np.array([[0.01], [0.02]])
    linksLengths = [1.0, 1.0]
    COMs = [0.5, 0.5]
    mass = [1.0, 1.0]
    inertia = [np.eye(3), np.eye(3)]

    print("=== Symbolic DH Test ===")
    robot_sym = Serial(
        jointsPositions,
        jointsVelocities,
        jointsAccelerations,
        linksLengths,
        COMs,
        mass,
        inertia,
        name="SymRobot",
        user_DH=user_DH_sym,
        symbolic=True,
    )
    pprint(robot_sym.symbolicDHParameters)

    print("\n=== Numeric DH Test ===")
    robot_num = Serial(
        jointsPositions,
        jointsVelocities,
        jointsAccelerations,
        linksLengths,
        COMs,
        mass,
        inertia,
        name="NumRobot",
        user_DH=user_DH_num,
        symbolic=False,
    )
    print(robot_num.dhParameters)
    # still in your __main__ block, after the DH tests:
    print("Joint 1 is at", robot_sym.whereIsTheJoint(1))  # should be (1, 0)
    print("COM 2 is at", robot_sym.whereIsTheCOM(2))  # should be (2, 2)
