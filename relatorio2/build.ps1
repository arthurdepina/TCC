$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $script_dir

pandoc artigo.md `
  --reference-doc=reference.docx `
  --bibliography=referencias.bib `
  --csl=abnt.csl `
  --citeproc `
  -o artigo_final.docx

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK: artigo_final.docx gerado com sucesso."
    Start-Process artigo_final.docx
} else {
    Write-Host "ERRO: falha ao gerar o documento."
}
