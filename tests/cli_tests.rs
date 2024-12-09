use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::{json, Value};
use std::fs;
use tempfile::tempdir;

#[test]
fn test_table_command() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("mina-zkml-cli")?;
    cmd.arg("table")
        .arg("-m")
        .arg("models/simple_perceptron.onnx");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Operator"))
        .stdout(predicate::str::contains("Input Shape"))
        .stdout(predicate::str::contains("Output Shape"));

    Ok(())
}

#[test]
fn test_convert_command() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    let output_path = temp_dir.path().join("model.json");

    let mut cmd = Command::cargo_bin("mina-zkml-cli")?;
    cmd.arg("convert")
        .arg("-m")
        .arg("models/simple_perceptron.onnx")
        .arg("-o")
        .arg(&output_path);

    cmd.assert().success();

    // Verify the output file exists and contains valid JSON
    let content = fs::read_to_string(&output_path)?;
    let json: Value = serde_json::from_str(&content)?;
    assert!(json.is_object());

    Ok(())
}

#[test]
fn test_proof_command() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;

    // Create test input file
    let input_path = temp_dir.path().join("input.json");
    let input_data = json!([[1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]);
    fs::write(&input_path, input_data.to_string())?;

    // Create output path for proof
    let output_path = temp_dir.path().join("proof.json");

    let mut cmd = Command::cargo_bin("mina-zkml-cli")?;
    cmd.arg("proof")
        .arg("-m")
        .arg("models/simple_perceptron.onnx")
        .arg("-i")
        .arg(&input_path)
        .arg("-o")
        .arg(&output_path)
        .arg("--input-visibility")
        .arg("public")
        .arg("--output-visibility")
        .arg("public");

    cmd.assert().success();

    // Verify the proof file exists and contains valid JSON
    let content = fs::read_to_string(&output_path)?;
    let json: Value = serde_json::from_str(&content)?;

    // Check required fields
    assert!(json["metadata"].is_object());
    assert!(json["metadata"]["model"].is_string());
    assert!(json["metadata"]["timestamp"].is_string());
    assert!(json["metadata"]["visibility"].is_object());
    assert!(json["input_shape"].is_array());
    assert!(json["proof"].is_string());

    Ok(())
}

#[test]
fn test_verify_command() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;

    // Create test input file
    let input_path = temp_dir.path().join("input.json");
    let input_data = json!([[1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]);
    fs::write(&input_path, input_data.to_string())?;

    // Generate proof first
    let proof_path = temp_dir.path().join("proof.json");
    let mut cmd = Command::cargo_bin("mina-zkml-cli")?;
    cmd.arg("proof")
        .arg("-m")
        .arg("models/simple_perceptron.onnx")
        .arg("-i")
        .arg(&input_path)
        .arg("-o")
        .arg(&proof_path)
        .arg("--input-visibility")
        .arg("public")
        .arg("--output-visibility")
        .arg("public");
    cmd.assert().success();

    // Read the proof file to get the output
    let proof_content = fs::read_to_string(&proof_path)?;
    let proof_json: Value = serde_json::from_str(&proof_content)?;

    // Create output file with the model's output
    let output_path = temp_dir.path().join("output.json");
    if let Some(output) = proof_json["output"].as_array() {
        fs::write(&output_path, serde_json::to_string(&output)?)?;
    } else {
        return Err("No output found in proof file".into());
    }

    // Now verify the proof with both input and output
    let mut cmd = Command::cargo_bin("mina-zkml-cli")?;
    cmd.arg("verify")
        .arg("-p")
        .arg(&proof_path)
        .arg("-m")
        .arg("models/simple_perceptron.onnx")
        .arg("-i")
        .arg(&input_path)
        .arg("--output")
        .arg(&output_path)
        .arg("--input-visibility")
        .arg("public")
        .arg("--output-visibility")
        .arg("public");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("✅ Proof verification successful"));

    Ok(())
}

#[test]
fn test_show_proof_command() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;

    // Create test input file
    let input_path = temp_dir.path().join("input.json");
    let input_data = json!([[1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]);
    fs::write(&input_path, input_data.to_string())?;

    // Generate proof first
    let proof_path = temp_dir.path().join("proof.json");
    let mut cmd = Command::cargo_bin("mina-zkml-cli")?;
    cmd.arg("proof")
        .arg("-m")
        .arg("models/simple_perceptron.onnx")
        .arg("-i")
        .arg(&input_path)
        .arg("-o")
        .arg(&proof_path)
        .arg("--input-visibility")
        .arg("public")
        .arg("--output-visibility")
        .arg("public");
    cmd.assert().success();

    // Now show the proof details
    let mut cmd = Command::cargo_bin("mina-zkml-cli")?;
    cmd.arg("show-proof").arg("-p").arg(&proof_path);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Property"))
        .stdout(predicate::str::contains("Value"))
        .stdout(predicate::str::contains("Input Shape"));

    Ok(())
}

#[test]
fn test_invalid_input_json() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;

    // Create invalid input file (not an array of arrays)
    let input_path = temp_dir.path().join("invalid_input.json");
    let invalid_input = json!({ "data": [1.0, 2.0] });
    fs::write(&input_path, invalid_input.to_string())?;

    let output_path = temp_dir.path().join("proof.json");

    let mut cmd = Command::cargo_bin("mina-zkml-cli")?;
    cmd.arg("proof")
        .arg("-m")
        .arg("models/simple_perceptron.onnx")
        .arg("-i")
        .arg(&input_path)
        .arg("-o")
        .arg(&output_path)
        .arg("--input-visibility")
        .arg("public")
        .arg("--output-visibility")
        .arg("public");

    cmd.assert().failure().stderr(predicate::str::contains(
        "Input JSON must be an array of arrays",
    ));

    Ok(())
}

#[test]
fn test_invalid_visibility() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;

    let input_path = temp_dir.path().join("input.json");
    let input_data = json!([[1.0, 2.0]]);
    fs::write(&input_path, input_data.to_string())?;

    let output_path = temp_dir.path().join("proof.json");

    let mut cmd = Command::cargo_bin("mina-zkml-cli")?;
    cmd.arg("proof")
        .arg("-m")
        .arg("models/simple_perceptron.onnx")
        .arg("-i")
        .arg(&input_path)
        .arg("-o")
        .arg(&output_path)
        .arg("--input-visibility")
        .arg("invalid") // Invalid visibility value
        .arg("--output-visibility")
        .arg("public");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Invalid visibility"));

    Ok(())
}
