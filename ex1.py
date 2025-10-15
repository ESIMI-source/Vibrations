import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def analyse_vibrations(omega_0, x0, v0, xi_sous=0.2, xi_suramorti=1.5, t_max=8):
    """
    Analyse des vibrations pour différents taux d'amortissement
    """
    # Domaine temporel
    t = np.linspace(0, t_max, 2000)
    
    # Cas 1: Sans amortissement (ξ = 0)
    xi_sans = 0.0
    omega_sans = omega_0 * np.sqrt(1 - xi_sans**2)
    x_sans = x0 * np.cos(omega_sans * t) + (v0/omega_sans) * np.sin(omega_sans * t)
    
    # Cas 2: Sous-amorti (0 < ξ < 1)
    omega_d_sous = omega_0 * np.sqrt(1 - xi_sous**2)
    x_sous = np.exp(-xi_sous * omega_0 * t) * (
        x0 * np.cos(omega_d_sous * t) + 
        (v0 + x0 * xi_sous * omega_0) / omega_d_sous * np.sin(omega_d_sous * t)
    )
    enveloppe_sous_pos = np.exp(-xi_sous * omega_0 * t) * np.sqrt(x0**2 + ((v0 + x0 * xi_sous * omega_0)/omega_d_sous)**2)
    enveloppe_sous_neg = -enveloppe_sous_pos
    
    # Cas 3: Amortissement critique (ξ = 1)
    xi_critique = 1.0
    x_critique = np.exp(-omega_0 * t) * (x0 * (1 + omega_0 * t) + v0 * t)
    
    # Cas 4: Sur-amorti (ξ > 1)
    omega_d_sur = omega_0 * np.sqrt(xi_suramorti**2 - 1)
    x_suramorti = np.exp(-xi_suramorti * omega_0 * t) * (
        x0 * np.cosh(omega_d_sur * t) + 
        (v0 + x0 * xi_suramorti * omega_0) / omega_d_sur * np.sinh(omega_d_sur * t)
    )
    
    # Calcul des amplitudes maximales pour l'échelle des axes
    amplitude_max = max(
        np.max(np.abs(x_sans)),
        np.max(np.abs(x_sous)),
        np.max(np.abs(x_critique)),
        np.max(np.abs(x_suramorti))
    )
    
    return t, x_sans, x_sous, x_critique, x_suramorti, enveloppe_sous_pos, enveloppe_sous_neg, amplitude_max

def create_plot(t, x_sans, x_sous, x_critique, x_suramorti, enveloppe_sous_pos, enveloppe_sous_neg, 
                amplitude_max, t_max, omega_0, x0, v0, xi_sous, xi_suramorti, zoom=False):
    """Crée le graphique avec ou sans zoom"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    xi_sans = 0.0
    xi_critique = 1.0
    
    ax.plot(t, x_sans, 'g-', linewidth=2.5, label=f'Sans amortissement (ξ = {xi_sans})')
    ax.plot(t, x_sous, 'm-', linewidth=2, label=f'Sous-amorti (ξ = {xi_sous})')
    ax.plot(t, x_critique, 'b-', linewidth=2, label=f'Amortissement critique (ξ = {xi_critique})')
    ax.plot(t, x_suramorti, 'r-', linewidth=2, label=f'Sur-amorti (ξ = {xi_suramorti})')
    
    # Enveloppes pour le cas sous-amorti
    ax.plot(t, enveloppe_sous_pos, 'm--', linewidth=1, alpha=0.7, label='Enveloppe exponentielle')
    ax.plot(t, enveloppe_sous_neg, 'm--', linewidth=1, alpha=0.7)
    
    # Ligne zéro
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    if not zoom:
        # Annotations simplifiées sans LaTeX complexe
        y_range = 2.0 * amplitude_max
        y_annot_sans = 0.7 * y_range
        y_annot_sous = 0.5 * y_range
        y_annot_critique = 0.2 * y_range
        y_annot_sur = 0.3 * y_range
        
        # Annotations avec texte simple
        ax.annotate('Sans amortissement', 
                   xy=(0.8, x_sans[200]), xytext=(t_max*0.4, y_annot_sans),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        ax.annotate('Sous-amorti', 
                   xy=(1.2, x_sous[300]), xytext=(t_max*0.5, y_annot_sous),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", alpha=0.7))
        
        ax.annotate('Amortissement critique', 
                   xy=(0.6, x_critique[150]), xytext=(t_max*0.3, y_annot_critique),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        ax.annotate('Sur-amorti', 
                   xy=(1.0, x_suramorti[250]), xytext=(t_max*0.6, y_annot_sur),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose", alpha=0.7))
    
    # Configuration du graphique
    ax.set_xlabel('Temps (s)', fontsize=12)
    ax.set_ylabel('Déplacement x(t) (m)', fontsize=12)
    title = 'Comportement initial (zoom)' if zoom else 'Solutions de l\'équation différentielle'
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Échelle adaptative
    y_margin = 0.1 * amplitude_max
    ax.set_ylim(-amplitude_max - y_margin, amplitude_max + y_margin)
    
    if zoom:
        ax.set_xlim(0, min(3, t_max))
    
    return fig

def main():
    st.set_page_config(page_title="Analyse des Vibrations", page_icon="📈", layout="wide")
    
    st.title("📈 Analyse des Vibrations Mécaniques")
    st.markdown(
        "Cette application visualise les solutions de l'équation différentielle pour différents taux d'amortissement. "
        "Ajustez les paramètres ci-dessous pour voir l'effet sur le comportement vibratoire."
    )
    
    # Sidebar avec les paramètres
    st.sidebar.header("Paramètres d'entrée")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        omega_0 = st.slider("Fréquence naturelle ω₀ (rad/s)", 
                           min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        x0 = st.slider("Déplacement initial x₀ (m)", 
                      min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    
    with col2:
        v0 = st.slider("Vitesse initiale ẋ₀ (m/s)", 
                      min_value=-5.0, max_value=5.0, value=0.5, step=0.1)
        t_max = st.slider("Temps maximum (s)", 
                         min_value=2, max_value=20, value=8, step=1)
    
    st.sidebar.header("Paramètres d'amortissement")
    xi_sous = st.sidebar.slider("Taux sous-amorti ξ", 
                               min_value=0.01, max_value=0.99, value=0.2, step=0.01)
    xi_suramorti = st.sidebar.slider("Taux sur-amorti ξ", 
                                    min_value=1.01, max_value=3.0, value=1.5, step=0.1)
    
    # Calcul des résultats
    t, x_sans, x_sous, x_critique, x_suramorti, enveloppe_sous_pos, enveloppe_sous_neg, amplitude_max = analyse_vibrations(
        omega_0, x0, v0, xi_sous, xi_suramorti, t_max
    )
    
    # Calcul des paramètres importants
    omega_d_sous = omega_0 * np.sqrt(1 - xi_sous**2)
    omega_d_sur = omega_0 * np.sqrt(xi_suramorti**2 - 1)
    
    # Affichage des résultats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Graphique complet")
        fig_complet = create_plot(t, x_sans, x_sous, x_critique, x_suramorti, 
                                 enveloppe_sous_pos, enveloppe_sous_neg, 
                                 amplitude_max, t_max, omega_0, x0, v0, xi_sous, xi_suramorti, zoom=False)
        st.pyplot(fig_complet)
        plt.close(fig_complet)
    
    with col2:
        st.subheader("Comportement initial (zoom)")
        fig_zoom = create_plot(t, x_sans, x_sous, x_critique, x_suramorti, 
                              enveloppe_sous_pos, enveloppe_sous_neg, 
                              amplitude_max, t_max, omega_0, x0, v0, xi_sous, xi_suramorti, zoom=True)
        st.pyplot(fig_zoom)
        plt.close(fig_zoom)
    
    # Section des paramètres calculés
    st.subheader("📊 Paramètres Calculés")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Amplitude maximale", f"{amplitude_max:.3f} m")
        st.metric("Fréquence naturelle", f"{omega_0:.2f} rad/s")
    
    with col2:
        st.metric("Période sous-amortie", f"{2*np.pi/omega_d_sous:.3f} s")
        st.metric("ω_d sous-amorti", f"{omega_d_sous:.3f} rad/s")
    
    with col3:
        st.metric("ω_d sur-amorti", f"{omega_d_sur:.3f} rad/s")
        st.metric("Décroissance critique", f"{omega_0:.2f}")
    
    # Explication théorique
    with st.expander("📚 Explication théorique"):
        st.markdown("""
**Équations différentielles résolues :**

- **Sans amortissement (ξ = 0):**  
x(t) = x₀·cos(ω₀t) + (ẋ₀/ω₀)·sin(ω₀t)

- **Sous-amorti (0 < ξ < 1):**  
x(t) = e^(-ξω₀t)·[x₀·cos(ω_d t) + (ẋ₀ + x₀ξω₀)/ω_d·sin(ω_d t)]  
avec ω_d = ω₀√(1-ξ²)

- **Amortissement critique (ξ = 1):**  
x(t) = e^(-ω₀t)·[x₀(1 + ω₀t) + ẋ₀t]

- **Sur-amorti (ξ > 1):**  
x(t) = e^(-ξω₀t)·[x₀·cosh(ω_d t) + (ẋ₀ + x₀ξω₀)/ω_d·sinh(ω_d t)]  
avec ω_d = ω₀√(ξ²-1)
        """)
    
    # Affichage des équations dans un format simple
    with st.expander("🧮 Équations détaillées"):
        st.write("**Sans amortissement:**")
        st.latex(r"x(t) = x_0 \cos(\omega_0 t) + \frac{\dot{x}_0}{\omega_0} \sin(\omega_0 t)")
        
        st.write("**Sous-amorti:**")
        st.latex(r"x(t) = e^{-\xi\omega_0 t} \left[x_0 \cos(\omega_d t) + \frac{\dot{x}_0 + x_0\xi\omega_0}{\omega_d} \sin(\omega_d t)\right]")
        st.latex(r"\omega_d = \omega_0 \sqrt{1 - \xi^2}")
        
        st.write("**Amortissement critique:**")
        st.latex(r"x(t) = e^{-\omega_0 t} \left[x_0 (1 + \omega_0 t) + \dot{x}_0 t\right]")
        
        st.write("**Sur-amorti:**")
        st.latex(r"x(t) = e^{-\xi\omega_0 t} \left[x_0 \cosh(\omega_d t) + \frac{\dot{x}_0 + x_0\xi\omega_0}{\omega_d} \sinh(\omega_d t)\right]")
        st.latex(r"\omega_d = \omega_0 \sqrt{\xi^2 - 1}")
    
    # Téléchargement des graphiques
    st.sidebar.header("Téléchargement")
    if st.sidebar.button("📥 Sauvegarder les graphiques"):
        fig_complet = create_plot(t, x_sans, x_sous, x_critique, x_suramorti, 
                                 enveloppe_sous_pos, enveloppe_sous_neg, 
                                 amplitude_max, t_max, omega_0, x0, v0, xi_sous, xi_suramorti, zoom=False)
        fig_zoom = create_plot(t, x_sans, x_sous, x_critique, x_suramorti, 
                              enveloppe_sous_pos, enveloppe_sous_neg, 
                              amplitude_max, t_max, omega_0, x0, v0, xi_sous, xi_suramorti, zoom=True)
        
        fig_complet.savefig("vibrations_complet.png", dpi=300, bbox_inches='tight')
        fig_zoom.savefig("vibrations_zoom.png", dpi=300, bbox_inches='tight')
        plt.close('all')
        
        st.sidebar.success("Graphiques sauvegardés!")

if __name__ == "__main__":
    main()
