#ifndef SINGLE_POSITION_KALMAN_FILTER_H
#define SINGLE_POSITION_KALMAN_FILTER_H


class SinglePositionKalmanFilter
{
    float m_muX;
    float m_muV;

    float m_sigmaXX;
    float m_sigmaVV;

    float m_qX;
    float m_qV;

public:
    SinglePositionKalmanFilter(
        float initialPosition,
        float initialVelocity,
        float initialPositionVariance,
        float initialVelocityVariance,
        float qPosition,
        float qVelocity);
    void update(float newPosition, float rPosition, float dt);
    [[nodiscard]] float position() const;
};

inline float SinglePositionKalmanFilter::position() const
{
    return m_muX;
}

#endif
