"""Named MediaPipe Face Mesh landmark indices used by notkoves.

Indices refer to the 478-point MediaPipe FaceLandmarker mesh (468 face points
+ 10 iris points). "R"/"L" are the *subject's* right/left, so R appears on the
image's left in a non-mirrored photo.

The mesh is a dense surface fit, not a set of anthropometric points, so these
are the closest mesh vertices to the classic soft-tissue landmarks (Farkas).
Expect a few pixels of systematic offset versus a hand-placed landmark.
"""

# --- Midline -----------------------------------------------------------------
FOREHEAD_TOP = 10      # top of the mesh. NOT the hairline (trichion).
GLABELLA = 9           # between the brows
NASION = 168           # bridge of the nose, between the eyes
PRONASALE = 1          # nose tip
SUBNASALE = 2          # base of the columella, where nose meets lip
LABRALE_SUP = 0        # top of upper-lip vermilion, midline
STOMION_UP = 13        # inner edge of upper lip
STOMION_LO = 14        # inner edge of lower lip
LABRALE_INF = 17       # bottom of lower-lip vermilion, midline
MENTON = 152           # bottom of the chin

MIDLINE = [FOREHEAD_TOP, GLABELLA, NASION, PRONASALE, SUBNASALE,
           LABRALE_SUP, STOMION_UP, STOMION_LO, LABRALE_INF, MENTON]

# --- Eyes --------------------------------------------------------------------
R_EX, R_EN = 33, 133          # right eye outer (exocanthion) / inner (endocanthion)
L_EN, L_EX = 362, 263         # left eye inner / outer
R_LID_UP, R_LID_LO = 159, 145
L_LID_UP, L_LID_LO = 386, 374

R_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                 173, 157, 158, 159, 160, 161, 246]
L_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                 466, 388, 387, 386, 385, 384, 398]

R_IRIS_CENTRE, R_IRIS_RING = 468, [469, 470, 471, 472]
L_IRIS_CENTRE, L_IRIS_RING = 473, [474, 475, 476, 477]

# --- Brows -------------------------------------------------------------------
R_BROW_OUT, R_BROW_PEAK, R_BROW_IN = 70, 105, 107
L_BROW_IN, L_BROW_PEAK, L_BROW_OUT = 336, 334, 300

# --- Nose / mouth / face outline ----------------------------------------------
R_ALA, L_ALA = 129, 358       # outer nostril wings (approximate alare)
R_CHEILION, L_CHEILION = 61, 291  # mouth corners
R_ZYGION, L_ZYGION = 234, 454     # widest point of the face outline (cheekbones)
R_GONION, L_GONION = 172, 397     # jaw angle (approximate; weak on soft jawlines)

# --- Bilateral pairs for symmetry, grouped by region ---------------------------
SYMMETRY_PAIRS = {
    "Eyes": [(33, 263), (133, 362), (159, 386), (145, 374)],
    "Brows": [(70, 300), (105, 334), (107, 336)],
    "Nose": [(129, 358), (98, 327)],
    "Mouth": [(61, 291), (37, 267), (84, 314)],
    "Cheeks": [(234, 454), (50, 280), (123, 352)],
    "Jaw & chin": [(172, 397), (136, 365), (150, 379), (176, 400)],
}
