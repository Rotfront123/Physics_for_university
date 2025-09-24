import numpy as np
import matplotlib.pyplot as plt

def simulate_trajectory(model_type, v0, angle, k=0.0, dt=0.01, max_time=20):
    g = 9.81
    angle_rad = np.radians(angle)
    x, y = 0, 0
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    
    x_points, y_points = [x], [y]
    t_points = [0]
    t = 0
    while y >= 0 and t < max_time:
        v = np.sqrt(vx**2 + vy**2)
        if model_type == 'linear':
            F_res = k * v
        else:
            F_res = k * v**2
        ax = -F_res * vx / v if v > 0 else 0
        ay = -g - F_res * vy / v if v > 0 else -g
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
        
        x_points.append(x)
        y_points.append(y)
        t_points.append(t)
    
    return x_points, y_points, t_points, x_points[-1]

def theoretical_for_linear(v0, angle, k=0.0, m=1.0, g=9.81):
    angle_rad = np.radians(angle)
    gamma = k / m
    
    if k == 0: 
        range_theor = (v0**2 * np.sin(2 * angle_rad)) / g
        height_theor = (v0**2 * np.sin(angle_rad)**2) / (2 * g)
        time_theor = (2 * v0 * np.sin(angle_rad)) / g
    else:
        def y_func(t):
            return (m/k) * (v0 * np.sin(angle_rad) + (m*g)/k) * (1 - np.exp(-gamma * t)) - (m*g/k) * t
        
        t_min, t_max = 0, 10 
        for _ in range(50):
            t_mid = (t_min + t_max) / 2
            if y_func(t_mid) > 0:
                t_min = t_mid
            else:
                t_max = t_mid
            if t_max - t_min < 1e-6:
                break
        time_theor = (t_min + t_max) / 2
        range_theor = (v0 * np.cos(angle_rad) * m / k) * (1 - np.exp(-gamma * time_theor))
        
        t_max_height = (m/k) * np.log(1 + (k * v0 * np.sin(angle_rad)) / (m*g))
        height_theor = y_func(t_max_height)
    
    return range_theor, height_theor, time_theor

def plot_angle_comparison():
    """Режим 1: Влияние угла на бросок"""
    print("\n=== Режим: Влияние угла на бросок ===")
    
    try:
        v0 = float(input("Начальная скорость (м/с) [20]: ") or 20)
        k_linear = float(input("Коэффициент линейного сопротивления [0.1]: ") or 0.1)
        k_quadratic = float(input("Коэффициент квадратичного сопротивления [0.01]: ") or 0.01)
    except ValueError:
        print("Ошибка ввода, используем значения по умолчанию")
        v0, k_linear, k_quadratic = 20, 0.1, 0.01
    
    angles = [30, 45, 60, 75]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    for angle in angles:
        x, y, t, fall_point = simulate_trajectory('linear', v0, angle, k_linear)
        ax1.plot(x, y, label=f'{angle}°', linewidth=2)
        ax2.plot(t[:-1], y[:-1], label=f'{angle}°', linewidth=2)
    
    for angle in angles:
        x, y, t, fall_point = simulate_trajectory('quadratic', v0, angle, k_quadratic)
        ax3.plot(x, y, label=f'{angle}°', linewidth=2)
        ax4.plot(t[:-1], y[:-1], label=f'{angle}°', linewidth=2)
    
    ax1.set_title(f'Линейное сопротивление (k={k_linear})')
    ax1.set_xlabel('Расстояние, м')
    ax1.set_ylabel('Высота, м')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_title(f'Линейное сопротивление (k={k_linear})')
    ax2.set_xlabel('Время, с')
    ax2.set_ylabel('Высота, м')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3.set_title(f'Квадратичное сопротивление (k={k_quadratic})')
    ax3.set_xlabel('Расстояние, м')
    ax3.set_ylabel('Высота, м')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4.set_title(f'Квадратичное сопротивление (k={k_quadratic})')
    ax4.set_xlabel('Время, с')
    ax4.set_ylabel('Высота, м')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    print("\nРезультаты для квадратичного сопротивления:")
    for angle in angles:
        x, y, t, fall_point = simulate_trajectory('quadratic', v0, angle, k_quadratic)
        range_theor, height_theor, time_theor = theoretical_for_linear(v0, angle, 0)
        print(f"Угол {angle}°: дальность = {fall_point:.1f} м (теор. без сопр. {range_theor:.1f} м), "
              f"высота = {max(y):.1f} м (теор. без сопр.{height_theor:.1f} м)")
    print("\nРезультаты для линейного сопротивления:")
    for angle in angles:
        x, y, t, fall_point = simulate_trajectory('linear', v0, angle, k_linear)
        range_theor, height_theor, time_theor = theoretical_for_linear(v0, angle, k_linear)
        print(f"Угол {angle}°: дальность = {fall_point:.1f} м (теор. линейное сопр. {range_theor:.1f} м), "
              f"высота = {max(y):.1f} м (теор. линейное сопр. {height_theor:.1f} м)")

def plot_velocity_comparison():
    """Режим 1: Влияние скорости на бросок"""
    print("\n=== Режим: Влияние скорости на бросок ===")
    
    try:
        angle = float(input("Угол броска (градусы) [45]: ") or 45)
        k_linear = float(input("Коэффициент линейного сопротивления [0.1]: ") or 0.1)
        k_quadratic = float(input("Коэффициент квадратичного сопротивления [0.01]: ") or 0.01)
    except ValueError:
        print("Ошибка ввода, используем значения по умолчанию")
        angle, k_linear, k_quadratic = 45, 0.1, 0.01
    
    velocities = [10, 20, 30, 40]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    for v0 in velocities:
        x, y, t, fall_point = simulate_trajectory('linear', v0, angle, k_linear)
        ax1.plot(x, y, label=f'{v0} м/с', linewidth=2)
        ax2.plot(t[:-1], y[:-1], label=f'{v0} м/с', linewidth=2)
    
    for v0 in velocities:
        x, y, t, fall_point = simulate_trajectory('quadratic', v0, angle, k_quadratic)
        ax3.plot(x, y, label=f'{v0} м/с', linewidth=2)
        ax4.plot(t[:-1], y[:-1], label=f'{v0} м/с', linewidth=2)
    
    ax1.set_title(f'Линейное сопротивление (k={k_linear})')
    ax1.set_xlabel('Расстояние, м')
    ax1.set_ylabel('Высота, м')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_title(f'Линейное сопротивление (k={k_linear})')
    ax2.set_xlabel('Время, с')
    ax2.set_ylabel('Высота, м')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3.set_title(f'Квадратичное сопротивление (k={k_quadratic})')
    ax3.set_xlabel('Расстояние, м')
    ax3.set_ylabel('Высота, м')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4.set_title(f'Квадратичное сопротивление (k={k_quadratic})')
    ax4.set_xlabel('Время, с')
    ax4.set_ylabel('Высота, м')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nРезультаты для квадратичного сопротивления:")
    for v0 in velocities:
        x, y, t, fall_point = simulate_trajectory('quadratic', v0, angle, k_quadratic)
        range_theor, height_theor, time_theor = theoretical_for_linear(v0, angle, 0)  # Теория без сопротивления
        print(f"Скорость {v0} м/с: дальность = {fall_point:.1f} м (теор. без сопр. {range_theor:.1f} м), "
              f"высота = {max(y):.1f} м (теор. без сопр. {height_theor:.1f} м)")
    print("\nРезультаты для линейного сопротивления:")
    for v0 in velocities:
        x, y, t, fall_point = simulate_trajectory('linear', v0, angle, k_linear)
        range_theor, height_theor, time_theor = theoretical_for_linear(v0, angle, k_linear)
        print(f"Скорость {v0} м/с: дальность = {fall_point:.1f} м (теор. линейное сопротивление {range_theor:.1f} м), "
              f"высота = {max(y):.1f} м (теор. линейное сопротивление {height_theor:.1f} м)")
    

def personal_simulation():
    """Режим 2: Персональное моделирование"""
    print("\n=== Режим: Персональное моделирование ===")
    
    try:
        v0 = float(input("Начальная скорость (м/с): "))
        angle = float(input("Угол броска (градусы): "))
        k_linear = float(input("Коэффициент линейного сопротивления [0.1]: ") or 0.1)
        k_quadratic = float(input("Коэффициент квадратичного сопротивления [0.01]: ") or 0.01)
    except ValueError:
        print("Ошибка ввода!")
        return
    
    # Теория для случая без сопротивления и с линейным сопротивлением
    range_theor_no_res, height_theor_no_res, time_theor_no_res = theoretical_for_linear(v0, angle, 0)
    range_theor_linear, height_theor_linear, time_theor_linear = theoretical_for_linear(v0, angle, k_linear)
    
    x_linear, y_linear, t_linear, fall_linear = simulate_trajectory('linear', v0, angle, k_linear)
    x_quadratic, y_quadratic, t_quadratic, fall_quadratic = simulate_trajectory('quadratic', v0, angle, k_quadratic)
    x_no_res, y_no_res, t_no_res, fall_no_res = simulate_trajectory('linear', v0, angle, 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(x_linear, y_linear, 'b-', label=f'Линейное (k={k_linear})', linewidth=2)
    ax1.plot(x_quadratic, y_quadratic, 'r-', label=f'Квадратичное (k={k_quadratic})', linewidth=2)
    ax1.plot(x_no_res, y_no_res, 'g--', label='Без сопротивления', linewidth=2)
    ax1.plot(fall_linear, 0, 'bo', markersize=8)
    ax1.plot(fall_quadratic, 0, 'ro', markersize=8)
    ax1.plot(fall_no_res, 0, 'go', markersize=8)
    
    ax1.set_xlabel('Расстояние, м')
    ax1.set_ylabel('Высота, м')
    ax1.set_title('Сравнение траекторий')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    ax2.plot(t_linear[:-1], y_linear[:-1], 'b-', 
             label=f'Линейное (k={k_linear})', linewidth=2)
    ax2.plot(t_quadratic[:-1], y_quadratic[:-1], 'r-', 
             label=f'Квадратичное (k={k_quadratic})', linewidth=2)
    ax2.plot(t_no_res[:-1], y_no_res[:-1], 'g--', 
             label='Без сопротивления', linewidth=2)
    
    ax2.set_xlabel('Время, с')
    ax2.set_ylabel('Высота, м')
    ax2.set_title('Зависимость высоты от времени')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

    print("\n=== РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ ===")
    print(f"Параметры: v0 = {v0} м/с, угол = {angle}°")
    print(f"Теоретически (без сопротивления):")
    print(f"  Дальность: {range_theor_no_res:.2f} м")
    print(f"  Макс. высота: {height_theor_no_res:.2f} м")
    print(f"  Время полета: {time_theor_no_res:.2f} с")
    
    print(f"Теоретически (линейное сопротивление k={k_linear}):")
    print(f"  Дальность: {range_theor_linear:.2f} м")
    print(f"  Макс. высота: {height_theor_linear:.2f} м")
    print(f"  Время полета: {time_theor_linear:.2f} с")
    
    print(f"\nЧисленное моделирование:")
    print(f"Линейное сопротивление (k={k_linear}):")
    print(f"  Дальность: {fall_linear:.2f} м ({fall_linear/range_theor_linear*100:.1f}% от теоретической)")
    print(f"  Макс. высота: {max(y_linear):.2f} м ({max(y_linear)/height_theor_linear*100:.1f}% от теоретической)")
    print(f"  Время полета: {t_linear[-1]:.2f} с")
    
    print(f"\nКвадратичное сопротивление (k={k_quadratic}):")
    print(f"  Дальность: {fall_quadratic:.2f} м ({fall_quadratic/range_theor_no_res*100:.1f}% от теоретической без сопр.)")
    print(f"  Макс. высота: {max(y_quadratic):.2f} м ({max(y_quadratic)/height_theor_no_res*100:.1f}% от теоретической без сопр.)")
    print(f"  Время полета: {t_quadratic[-1]:.2f} с")

def main():
    """Главное меню программы"""
    while True:
        print("\n" + "="*60)
        print("           МОДЕЛИРОВАНИЕ ДВИЖЕНИЯ КАМНЯ")
        print("="*60)
        print("\nРежим 1: Исследование параметров")
        print("  1.1 - Влияние угла на бросок")
        print("  1.2 - Влияние скорости на бросок")
        print("\nРежим 2: Персональное моделирование")
        print("  2 - Ввод пользовательских параметров")
        print("\n  0 - Выход")
        
        choice = input("\nВыберите режим: ").strip()
        
        if choice == '1.1':
            plot_angle_comparison()
        elif choice == '1.2':
            plot_velocity_comparison()
        elif choice == '2':
            personal_simulation()
        elif choice == '0':
            print("Выход из программы...")
            break
        else:
            print("Неверный выбор, попробуйте снова")

if __name__ == "__main__":
    main()
