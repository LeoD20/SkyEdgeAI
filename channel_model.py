import numpy as np
import bisect
from scipy import constants


# ------------------------------
# dBm and Watt Conversion
# ------------------------------

def dbm_to_watt(dbm_value):
    """
    Converts power from dBm to Watts.
    P(W) = 10^((P(dBm) - 30) / 10)
    """
    try:
        return 10 ** ((dbm_value - 30) / 10)
    except TypeError:
        raise TypeError(f"Invalid input type for dBm: {dbm_value}")

def watt_to_dbm(watt_value):
    """
    Converts power from Watts to dBm.
    P(dBm) = 10 * log10(P(W)) + 30
    """
    if not isinstance(watt_value, (int, float)):
        raise TypeError(f"Watt_value must be a number, got {type(watt_value)}")
    if watt_value <= 0:
        raise ValueError(f"Watt_value must be > 0, got {watt_value}")
    return 10 * np.log10(watt_value) + 30


# ------------------------------
# Distance and Angle Computation
# ------------------------------

def compute_distance(object1, object2):
    """
    Calculates the Euclidean distance between two 3D points.

    Args:
        object1 (array-like): Coordinates of first point (length 3).
        object2 (array-like): Coordinates of second point (length 3).

    Returns:
        float: Euclidean distance.

    Raises:
        ValueError: If input vectors are not both 3D or have different shapes.
    """
    object1 = np.array(object1)
    object2 = np.array(object2)

    if object1.shape != object2.shape or object1.shape[0] != 3:
        raise ValueError("Both input vectors must be 3D and have the same shape.")

    return float(np.linalg.norm(object1 - object2))

def compute_elevation_angle(uav_position, user_position):
    """
    Computes the elevation angle from user to UAV in radians.

    Angle is measured from the horizontal plane to the UAV.
    Returns π/2 if UAV is directly above the user.

    Args:
        uav_position (array-like): UAV 3D coordinates.
        user_position (array-like): User 3D coordinates.

    Returns:
        float: Elevation angle in radians.
    """
    uav_position = np.array(uav_position)
    user_position = np.array(user_position)

    if uav_position.shape[0] != 3 or user_position.shape[0] != 3:
        raise ValueError("Both UAV and user positions must be 3D vectors.")

    delta_h = abs(uav_position[2] - user_position[2])
    horizontal_distance = np.linalg.norm(uav_position[:2] - user_position[:2])

    if np.isclose(horizontal_distance, 0.0):
        return np.pi / 2
    return np.arctan(delta_h / horizontal_distance)


# ------------------------------
# Free-space Path Loss
# ------------------------------

def compute_path_loss(distance, frequency):
    """
    Computes free-space path loss in the linear scale using Friis equation.

    The formula used is:
        PL_dB = 20*log10(d) + 20*log10(f) + 20*log10(4π / c)

    Args:
        distance (float): Distance between transmitter and receiver in meters.
        frequency (float): Carrier frequency in Hz.

    Returns:
        float: Path loss as a dimensionless linear scale ratio (not in dB).

    Raises:
        ValueError: If distance or frequency are non-positive.
    """
    if not isinstance(distance, (int, float)) or not isinstance(frequency, (int, float)):
        raise TypeError("Both distance and frequency must be numeric.")

    if distance <= 0:
        raise ValueError(f"Distance must be > 0, got {distance}")
    if frequency <= 0:
        raise ValueError(f"Frequency must be > 0, got {frequency}")

    # Compute path loss in dB
    const_term = 20 * np.log10(4 * np.pi / constants.c)
    path_loss_db = 20 * np.log10(distance) + 20 * np.log10(frequency) + const_term

    # Convert to linear scale
    return 10 ** (-path_loss_db / 10)


# ------------------------------
# LoS Probability
# ------------------------------

def compute_los_probability(theta_deg, a, b, c, d):
    """
    Computes the probability of Line-of-Sight (LoS) based on an elevation angle (in degrees).

    The probability is computed using a sigmoid function with empirically fitted coefficients:
        P_LoS = 1 / (1 + exp(a * θ³ + b * θ² + c * θ + d))

    Args:
        theta_deg (float): Elevation angle between user and UAV, in degrees.
        a (float): Coefficient for the θ³ (cubic) term.
        b (float): Coefficient for the θ² (quadratic) term.
        c (float): Coefficient for the θ (linear) term.
        d (float): Constant offset term in the polynomial.

    Returns:
        float: Probability of LoS (between 0 and 1).

    Raises:
        TypeError: If any input is not a numeric value.
    """
    if not isinstance(theta_deg, (int, float)):
        raise TypeError("theta_deg must be a numeric value.")
    for coeff in (a, b, c, d):
        if not isinstance(coeff, (int, float)):
            raise TypeError("All polynomial coefficients must be numeric.")

    e = a * theta_deg**3 + b * theta_deg**2 + c * theta_deg + d
    try:
        return 1 / (1 + np.exp(e))
    except OverflowError:
        return 0.0 if e > 0 else 1.0  # sigmoid asymptotes

def is_los(p_los):
    """
    Stochastically determines the presence of Line-of-Sight based on probability.

    Args:
        p_los (float): Probability of LoS, must be in [0, 1].

    Returns:
        bool: True if LoS is realized, False otherwise.

    Raises:
        ValueError: If p_los is outside [0, 1].
    """
    if not isinstance(p_los, (int, float)):
        raise TypeError("p_los must be numeric.")
    if not (0 <= p_los <= 1):
        raise ValueError(f"p_los must be between 0 and 1, got {p_los}")
    return np.random.rand() < p_los


# ------------------------------
# Rician Fading
# ------------------------------

def generate_rician_channel(K_factor, nlos_power=1.0):
    """
    Generates the magnitude of a Rician fading channel coefficient.

    The Rician model consists of:
        - A deterministic Line-of-Sight (LoS) component with random phase.
        - A stochastic Non-Line-of-Sight (NLoS) component modeled as complex Gaussian noise.

    This function returns the **fading amplitude**, defined as:
        |h| = |h_LOS + h_NLOS|

    To obtain **fading power gain**, which is needed when computing received power or SNR,
    use |h|².

    Args:
        K_factor (float): Rician K-factor (LoS-to-NLoS power ratio), must be ≥ 0.
        nlos_power (float): Average power of the NLoS component (default = 1.0).

    Returns:
        float: Fading amplitude (|h|). Square this to get power gain.

    Raises:
        TypeError: If inputs are not numeric.
        ValueError: If K_factor < 0 or nlos_power ≤ 0.
    """

    if not isinstance(K_factor, (int, float)):
        raise TypeError(f"K_factor must be a number, got {type(K_factor)}")
    if not isinstance(nlos_power, (int, float)):
        raise TypeError(f"nlos_power must be a number, got {type(nlos_power)}")
    if K_factor < 0:
        raise ValueError(f"K_factor must be >= 0, got {K_factor}")
    if nlos_power <= 0:
        raise ValueError(f"nlos_power must be > 0, got {nlos_power}")

    component_variance = nlos_power / 2  # Variance per dimension (real/imag)

    if K_factor == 0:
        # Rayleigh fading (pure NLoS)
        real = np.random.normal(0, np.sqrt(component_variance))
        imag = np.random.normal(0, np.sqrt(component_variance))
        return abs(real + 1j * imag)

    # Rician fading (LoS + NLoS)
    m = np.sqrt(2 * K_factor * component_variance)  # LoS component amplitude
    theta = np.random.uniform(0, 2 * np.pi)
    h_LOS = m * np.exp(1j * theta)

    nlos_real = np.random.normal(0, np.sqrt(component_variance))
    nlos_imag = np.random.normal(0, np.sqrt(component_variance))
    h_NLOS = nlos_real + 1j * nlos_imag

    return abs(h_LOS + h_NLOS)


# ------------------------------
# UAV Velocity Computation
# ------------------------------

def compute_uav_velocity(uav_positions, current_step, time_step):
    """
    Computes the UAV velocity vector using finite difference between consecutive positions.

    Assumes discrete position sampling with a constant time step. If the current step is the last one,
    it wraps around using modular indexing (looped trajectory). If UAV is stationary, the velocity will be zero.

    Args:
        uav_positions (list or array): List of UAV 3D positions (shape: [N, 3]).
        current_step (int): Index of the current UAV position in the trajectory.
        time_step (float): Time interval between successive positions (in seconds).

    Returns:
        np.ndarray: 3D velocity vector of the UAV at the given step.

    Raises:
        ValueError: If time_step is not positive or if uav_positions is too short.
    """
    if time_step <= 0:
        raise ValueError(f"time_step must be positive, got {time_step}")
    if len(uav_positions) < 2:
        raise ValueError("At least two UAV positions are required to compute velocity.")

    position_now = np.array(uav_positions[current_step % len(uav_positions)])
    position_next = np.array(uav_positions[(current_step + 1) % len(uav_positions)])
    velocity_vector = (position_next - position_now) / time_step
    return velocity_vector



# ------------------------------
# Cosine of a Theta Angle
# ------------------------------

def compute_cos_theta(uav_position, uav_velocity_vector, user_position):
    """
    Computes the cosine of the angle between the UAV velocity vector and
    the vector from the UAV to the user.

    This is used to project the UAV’s motion onto the user direction,
    which is needed for Doppler shift computation or trajectory adaptation.

    Args:
        uav_position (array-like): 3D position of the UAV.
        uav_velocity_vector (array-like): 3D velocity vector of the UAV.
        user_position (array-like): 3D position of the user.

    Returns:
        float:
            - cos(θ) ∈ [-1, 1] if both vectors are valid.
            - 0.0 if the UAV is stationary.
            - -np.inf as a sentinel if UAV and user positions overlap.
    """
    relative_position = user_position - uav_position
    norm_velocity = np.linalg.norm(uav_velocity_vector)
    if norm_velocity <= 0:
        return 0.0  # UAV stationary
    norm_relative = np.linalg.norm(relative_position)
    if norm_relative <= 0:
        return -np.inf  # Overlapping positions (sentinel value)

    cos_theta = np.dot(uav_velocity_vector, relative_position) / (norm_velocity * norm_relative)
    return cos_theta


# ------------------------------
# Doppler Shift
# ------------------------------

def compute_doppler_shift(time_step, current_step, uav_positions, user_position, frequency):
    """
    Computes the Doppler frequency shift due to UAV motion relative to a user.

    The Doppler shift is calculated as:
        f_d = (v / c) * f * cos(θ)

    where θ is the angle between the UAV's velocity vector and the line
    connecting the UAV to the user. If the cosine is invalid (e.g., user and
    UAV positions coincide), the angle is approximated using the next spatial step.

    Args:
        time_step (float): Time step between UAV positions [s].
        current_step (int): Current step index in UAV trajectory.
        uav_positions (list of array-like): List of 3D UAV positions.
        user_position (array-like): Fixed 3D user position.
        frequency (float): Carrier frequency [Hz].

    Returns:
        float: Doppler shift in Hz (can be positive or negative).

    Notes:
        - If cos(θ) ≤ -1 (e.g., due to overlap, sentinel value), the next spatial step
          is used to approximate the angle.
    """
    uav_velocity_vector = compute_uav_velocity(uav_positions, current_step, time_step)
    velocity_norm = np.linalg.norm(uav_velocity_vector)
    cos_theta = compute_cos_theta(
        uav_positions[(current_step + 1) % len(uav_positions)],
        uav_velocity_vector,
        user_position
    )

    if cos_theta <= -1:
        update_pos = (current_step + 1) % len(uav_positions)
        cos_theta = compute_cos_theta(
            uav_positions[update_pos],
            uav_velocity_vector,
            user_position
        )
    f_d0 = (velocity_norm / constants.c) * frequency
    doppler_frequency = f_d0 * cos_theta
    return doppler_frequency, f_d0


# ------------------------------
# Shadowing Gain
# ------------------------------

def shadowing(std_shadowing=8):
    """
    Generates a multiplicative log-normal shadowing factor.

    Models large-scale fading variations due to environmental obstructions.
    The shadowing is sampled as:
        shadow_factor = 10^(X / 10), where X ~ N(0, σ²)

    Args:
        std_shadowing (float): Standard deviation σ of the shadowing in dB. Takes values in [5db, 12dB].
        We used 8dB, being a typical value for macro cellular systems, which is a good approximation of our system.

    Returns:
        float: Shadowing multiplier (unitless, > 0).

    Raises:
        ValueError: If variance is negative.
        TypeError: If input is not a number.
    """
    if not isinstance(std_shadowing, (int, float)):
        raise TypeError(f"Variance must be numeric, got {type(std_shadowing)}")
    if std_shadowing < 0:
        raise ValueError(f"Variance must be non-negative, got {std_shadowing}")

    return 10 ** (np.random.normal(0, std_shadowing) / 10)


# ------------------------------
# Channel
# ------------------------------

def compute_channel(uav_positions, user_position, current_step, time_now, time_step, frequency, K_factor_constant, shadowing_variance=8):
    """
    Computes the total complex channel coefficient h(t) between a UAV and a ground user.

    The channel model includes:
        - Free-space path loss
        - Rician small-scale fading (magnitude only)
        - Log-normal shadowing
        - Doppler shift phase rotation
        - Stochastic LoS/NLoS selection based on the elevation angle

    Args:
        uav_positions (list of arrays): UAV trajectory positions in 3D.
        user_position (array-like): 3D coordinates of the ground user.
        current_step (int): Index of the current UAV position in the trajectory.
        time_now (float): Current simulation time (s).
        time_step (float): Time interval between UAV positions (s).
        frequency (float): Carrier frequency (Hz).
        K_factor_constant (float): Rician K-factor used for LoS links.
        shadowing_variance (float, optional): Shadowing variance in dB² (default = 8).

    Returns:
        complex: Total channel coefficient h(t). To compute channel power gain, use abs(h_total)².
    """
    uav_position = np.array(uav_positions[current_step % len(uav_positions)])
    user_position = np.array(user_position)

    distance = compute_distance(uav_position, user_position)
    doppler_freq,f_d0 = compute_doppler_shift(time_step, current_step, uav_positions, user_position, frequency)
    path_loss = compute_path_loss(distance, frequency)

    elevation_angle_radians = compute_elevation_angle(uav_position, user_position)
    elevation_angle_degrees = np.degrees(elevation_angle_radians)

    # Empirical LoS model parameters (urban environment)
    a = -2.397e-5
    b = 0.0034
    c = -0.1985
    d = 3.7876

    p_los = compute_los_probability(elevation_angle_degrees, a, b, c, d)
    K_factor = K_factor_constant if is_los(p_los) else 0

    h_rician = generate_rician_channel(K_factor)
    doppler_effect = np.exp(1j * 2 * np.pi * doppler_freq * time_now)
    h_shad = shadowing(shadowing_variance)

    h_total = np.sqrt(path_loss) * h_rician * doppler_effect * np.sqrt(h_shad)
    return h_total


# ------------------------------
# Signal-to-Noise Ratio (SNR)
# ------------------------------

def compute_snr(h_total, transmit_power_watt, noise_power_watt):
    """
    Computes the signal-to-noise ratio (SNR) in linear scale.

    Formula:
        SNR = (P_tx * |h|²) / P_noise

    Args:
        h_total (complex): Total channel coefficient (includes fading, path loss, etc.).
        transmit_power_watt (float): Transmit power in Watts.
        noise_power_watt (float): Noise power in Watts.

    Returns:
        float: SNR in linear scale (unitless).

    Raises:
        ValueError: If power values are non-positive.
    """
    if transmit_power_watt <= 0:
        raise ValueError(f"Transmit power must be > 0, got {transmit_power_watt}")
    if noise_power_watt <= 0:
        raise ValueError(f"Noise power must be > 0, got {noise_power_watt}")

    channel_gain = np.abs(h_total) ** 2
    snr_linear = (transmit_power_watt * channel_gain) / noise_power_watt
    return snr_linear


# ------------------------------
# Spectral Efficiency
# ------------------------------

def get_spectral_efficiency(snr):
    """
    Maps SNR (in dB) to spectral efficiency using a threshold-based scheme.

    Uses binary search to efficiently select the appropriate modulation and coding
    rate based on the given SNR value. The mapping is derived from a predefined
    Modulation and Coding Scheme (MCS) table (e.g., LTE or 5G NR).

    Args:
        snr (float): Signal-to-noise ratio in dB.

    Returns:
        float: Spectral efficiency in bits/s/Hz. Returns 0 if SNR is too low.
    """
    snr_thresholds = [-9.478, -6.658, -4.098, -1.798, 0.399, 2.424, 3.871, 6.367,
                      8.456, 10.266, 12.218, 14.122, 15.849, 17.806, 19.809]
    spectral_efficiencies = [0.15237, 0.2344, 0.377, 0.6016, 0.877, 1.1758, 1.4766, 1.9141,
                             2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547]

    index = bisect.bisect_right(snr_thresholds, snr) - 1
    if index >= 0:
        return spectral_efficiencies[index]
    return 0.0
