from typing import Any

import httpx


class CryptoPriceTool:
    """
    Outil permettant de recuperer le cours actuel des cryptomonnaies via CoinGecko.
    """

    name = "crypto_price"
    description = (
        "Recupere le prix actuel d'une cryptomonnaie en temps reel "
        "(ex: bitcoin, ethereum). "
        "Retourne le prix dans la devise demandee (USD par defaut)."
    )
    args_schema = {
        "coin_id": (
            "L'identifiant de la crypto sur CoinGecko "
            "(ex: 'bitcoin', 'ethereum', 'solana')."
        ),
        "vs_currency": (
            "La devise de conversion (ex: 'usd', 'eur', 'jpy'). Defaut: 'usd'."
        ),
    }
    return_direct = True
    trigger_words: tuple[str, ...] = (
        "prix",
        "cours",
        "valeur",
        "bitcoin",
        "btc",
        "eth",
        "crypto",
        "cryptomonnaie",
    )

    async def execute(self, **kwargs: Any) -> str:
        coin_id = str(kwargs.get("coin_id", "bitcoin")).lower().strip()
        vs_currency = str(kwargs.get("vs_currency", "usd")).lower().strip()

        # Utilisation de l'API publique simple de CoinGecko
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": coin_id, "vs_currencies": vs_currency}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if not data or coin_id not in data:
                    return f"Erreur : Impossible de trouver le cours pour '{coin_id}'."

                price = data[coin_id][vs_currency]
                return (
                    f"Le cours actuel de {coin_id.capitalize()} est de {price} "
                    f"{vs_currency.upper()}."
                )
        except Exception as e:
            return f"Echec de la recuperation du cours : {str(e)}"
