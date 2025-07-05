def save_spectral_analysis(results, filename):
    """
    Save spectral analysis results to a text file
    
    Args:
        results (dict): Results from analyze_spectral_properties
        filename (str): Name of the file to save results to
    """
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SPECTRAL ANALYSIS RESULTS\n")
        f.write("="*60 + "\n")
        
        for regime, data in results.items():
            f.write(f"\n{regime.upper().replace('_', '-')} MODEL (Hidden units: {data['model_size']}):\n")
            f.write("-" * 40 + "\n")
            
            for i, layer_name in enumerate(data['layer_names']):
                f.write(f"{layer_name}:\n")
                f.write(f"  Spectral Norm: {data['spectral_norms'][i]:.6f}\n")
                f.write(f"  Max Eigenvalue: {data['eigenvalues'][i][0]:.6f}\n")
                f.write(f"  Min Positive Eigenvalue: {data['eigenvalues'][i][1]:.6f}\n")
                f.write(f"  Condition Number: {data['condition_numbers'][i]:.6f}\n")
                f.write("\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("SUMMARY COMPARISON\n")
        f.write("="*60 + "\n")
        
        # Compare spectral norms
        f.write("\nSpectral Norms Comparison:\n")
        f.write("-" * 25 + "\n")
        under_norms = results['under_parameterized']['spectral_norms']
        over_norms = results['over_parameterized']['spectral_norms']
        
        for i in range(len(under_norms)):
            f.write(f"Layer {i+1}: Under={under_norms[i]:.6f}, Over={over_norms[i]:.6f}\n")
        
        # Compare condition numbers
        f.write("\nCondition Numbers Comparison:\n")
        f.write("-" * 28 + "\n")
        under_conditions = results['under_parameterized']['condition_numbers']
        over_conditions = results['over_parameterized']['condition_numbers']
        
        for i in range(len(under_conditions)):
            f.write(f"Layer {i+1}: Under={under_conditions[i]:.6f}, Over={over_conditions[i]:.6f}\n")
    
    print(f"\nSpectral analysis results saved to '{filename}'")
