"""
Exemple d'utilisation du générateur de morphings avancé
"""

from morph_advanced import MorphingConfig, MorphingEngine, ReportGenerator, ProgressTracker, setup_logging
from pathlib import Path

# ==================== Exemple 1: Configuration de base ====================

def example_quick_test():
    """Test rapide avec quelques morphings"""
    print("\n=== EXEMPLE 1: Test rapide ===\n")

    config = MorphingConfig(
        mode="sample",
        num_samples=10,  # Seulement 10 paires
        alpha_values=[0.5],  # Un seul alpha
        save_individual=True,
        save_html_report=True
    )

    logger = setup_logging(config.log_dir)
    engine = MorphingEngine(config, logger)
    engine.initialize()

    tracker = engine.run()
    tracker.print_summary()

    # Générer le rapport
    report_gen = ReportGenerator(config, tracker)
    html_file = report_gen.generate_html_report()
    print(f"\n✅ Rapport HTML: {html_file}")


# ==================== Exemple 2: Plusieurs alphas ====================

def example_multiple_alphas():
    """Génère plusieurs valeurs d'alpha pour voir la progression"""
    print("\n=== EXEMPLE 2: Plusieurs alphas ===\n")

    config = MorphingConfig(
        mode="per_person",  # Un morphing par personne
        alpha_values=[0.2, 0.4, 0.5, 0.6, 0.8],  # Progression douce
        save_individual=True,
        save_html_report=True,
        output_dir=Path("./morphing_results/multi_alpha")
    )

    logger = setup_logging(config.log_dir)
    engine = MorphingEngine(config, logger)
    engine.initialize()

    tracker = engine.run()
    tracker.print_summary()

    report_gen = ReportGenerator(config, tracker)
    html_file = report_gen.generate_html_report()
    csv_file = report_gen.generate_csv_report()

    print(f"\n✅ Rapports générés:")
    print(f"   HTML: {html_file}")
    print(f"   CSV: {csv_file}")


# ==================== Exemple 3: Batch de production ====================

def example_production_batch():
    """Configuration pour génération en masse"""
    print("\n=== EXEMPLE 3: Batch de production ===\n")

    config = MorphingConfig(
        mode="sample",
        num_samples=100,  # Plus grand batch
        alpha_values=[0.3, 0.5, 0.7],
        image_size=256,  # Images plus grandes
        save_individual=True,
        save_html_report=True,
        output_dir=Path("./morphing_results/production")
    )

    logger = setup_logging(config.log_dir)
    engine = MorphingEngine(config, logger)
    engine.initialize()

    tracker = engine.run()
    tracker.print_summary()

    # Analyse des résultats
    stats = tracker.get_stats()
    print(f"\n📊 Analyse des performances:")
    print(f"   - Taux de réussite: {stats['success_rate']:.1f}%")
    print(f"   - Vitesse moyenne: {stats['speed_per_sec']:.2f} morphings/sec")
    print(f"   - Temps total: {stats['elapsed_time']/60:.1f} minutes")

    # Sauvegarder les rapports
    report_gen = ReportGenerator(config, tracker)
    html_file = report_gen.generate_html_report()
    csv_file = report_gen.generate_csv_report()

    print(f"\n✅ Fichiers générés:")
    print(f"   - Images: {config.output_dir}")
    print(f"   - HTML: {html_file}")
    print(f"   - CSV: {csv_file}")
    print(f"   - Logs: {tracker.log_file}")


# ==================== Exemple 4: Configuration personnalisée ====================

def example_custom_config():
    """Exemple avec configuration personnalisée"""
    print("\n=== EXEMPLE 4: Configuration personnalisée ===\n")

    # Créer une configuration sur mesure
    config = MorphingConfig(
        # Dataset
        min_faces_per_person=50,  # Plus sélectif
        resize_factor=0.4,
        image_size=192,

        # Génération
        mode="sample",
        num_samples=30,
        alpha_values=[0.25, 0.5, 0.75],

        # Sauvegarde
        save_individual=True,
        save_html_report=True,

        # Chemins personnalisés
        output_dir=Path("./morphing_results/custom"),
        log_dir=Path("./morphing_logs/custom"),
        dlib_models_dir=Path("./dlib_models")
    )

    logger = setup_logging(config.log_dir)
    logger.info("Configuration personnalisée chargée")

    engine = MorphingEngine(config, logger)
    engine.initialize()

    tracker = engine.run()
    tracker.print_summary()

    # Analyse détaillée
    print("\n📈 Analyse détaillée:")
    successful_results = [r for r in tracker.results if r.success]

    if successful_results:
        avg_time = sum(r.processing_time for r in successful_results) / len(successful_results)
        print(f"   - Temps moyen par morphing: {avg_time:.3f}s")

        landmarks_detected = sum(1 for r in successful_results
                                if r.landmarks_detected_a and r.landmarks_detected_b)
        print(f"   - Landmarks détectés: {landmarks_detected}/{len(successful_results)} paires")

    # Générer les rapports
    report_gen = ReportGenerator(config, tracker)
    html_file = report_gen.generate_html_report()
    csv_file = report_gen.generate_csv_report()

    print(f"\n✅ Rapports:")
    print(f"   HTML: {html_file}")
    print(f"   CSV: {csv_file}")


# ==================== Menu principal ====================

def main():
    print("\n" + "="*70)
    print("🎭 EXEMPLES D'UTILISATION DU GÉNÉRATEUR DE MORPHINGS")
    print("="*70)
    print("\nChoisissez un exemple:")
    print("  1. Test rapide (10 morphings, 1 alpha)")
    print("  2. Plusieurs alphas (progression douce)")
    print("  3. Batch de production (100 morphings)")
    print("  4. Configuration personnalisée")
    print("  5. Tous les exemples")
    print("  0. Quitter")

    choice = input("\nVotre choix: ").strip()

    if choice == "1":
        example_quick_test()
    elif choice == "2":
        example_multiple_alphas()
    elif choice == "3":
        example_production_batch()
    elif choice == "4":
        example_custom_config()
    elif choice == "5":
        example_quick_test()
        example_multiple_alphas()
        example_production_batch()
        example_custom_config()
    elif choice == "0":
        print("\nAu revoir!")
        return
    else:
        print("\n❌ Choix invalide!")
        return

    print("\n" + "="*70)
    print("✅ TERMINÉ!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
