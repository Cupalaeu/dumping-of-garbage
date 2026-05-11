import json

# Arquivos
input_json = "dataset_index.json"
output_geojson = "mapa_lixos.geojson"

# Carrega JSON original
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# Estrutura GeoJSON
geojson = {
    "type": "FeatureCollection",
    "features": []
}

# Contadores para validação
total_validos = 0
total_adicionados = 0

# Converte pontos
for key, item in data.items():

    lat = item.get("latitude")
    lon = item.get("longitude")

    # Verifica se coordenadas existem e não são nulas
    if lat is None or lon is None:
        continue

    total_validos += 1

    feature = {
        "type": "Feature",
        "properties": {
            "id": key,
            "nome_original": item.get("nome_original"),
            "largura": item.get("largura"),
            "altura": item.get("altura")
        },
        "geometry": {
            "type": "Point",
            # GeoJSON usa [longitude, latitude]
            "coordinates": [lon, lat]
        }
    }

    geojson["features"].append(feature)
    total_adicionados += 1

# Salva GeoJSON
with open(output_geojson, "w", encoding="utf-8") as f:
    json.dump(geojson, f, ensure_ascii=False, indent=2)

print(f"GeoJSON salvo em: {output_geojson}")

# =========================
# VERIFICAÇÃO FINAL
# =========================

print("\n===== VERIFICAÇÃO =====")
print(f"Pontos válidos no JSON: {total_validos}")
print(f"Pontos adicionados ao GeoJSON: {total_adicionados}")

if total_validos == total_adicionados:
    print("✓ Todos os pontos não nulos foram adicionados corretamente.")
else:
    print("✗ ERRO: Existem pontos faltando no GeoJSON.")

    faltando = total_validos - total_adicionados
    print(f"Pontos faltando: {faltando}")