# ==============================
# NECESSÁRIO: API KEY do YouTube Data API v3
# ==============================

import os
import csv
import json
import time
from googleapiclient.errors import HttpError

# ==============================
# CONFIGURAÇÕES
# ==============================

VIDEOS_PER_CHANNEL = 5
MAX_COMMENTS_PER_VIDEO = 650

CHANNELS = {
    "Prazer, Karnal - Canal Oficial de Leandro Karnal": "profissional",
    "Rossandro Klinjey": "profissional",
    "PodPeople - Ana Beatriz Barbosa": "profissional",
    "Augusto Cury": "profissional",
    "Minutos Psíquicos": "profissional",
    "Casa do Saber": "profissional",

    "ellora": "amador",
    "JoutJout Prazer": "amador",
    "Ludoviajante": "amador",
    "Juliana Goes": "amador",
    "Fred Elboni": "amador",
    "Obvious": "amador",
}

# ==============================
# BUSCAR CHANNEL INFO
# ==============================

def get_channel_info(channel_name):

    res = youtube.search().list(
        q=channel_name,
        part="snippet",
        type="channel",
        maxResults=1
    ).execute()

    items = res.get("items")

    if not items:
        return None, None

    channel_id = items[0]["snippet"]["channelId"]

    res = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()

    uploads_playlist = res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    return channel_id, uploads_playlist


# ==============================
# METADADOS DOS VÍDEOS
# ==============================

def get_video_metadata(video_ids):

    metadata = {}

    for i in range(0, len(video_ids), 50):

        block = video_ids[i:i+50]

        res = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(block)
        ).execute()

        for item in res["items"]:

            vid = item["id"]

            snippet = item["snippet"]
            stats = item["statistics"]

            metadata[vid] = {
                "videoId": vid,
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "channelId": snippet.get("channelId"),
                "channelTitle": snippet.get("channelTitle"),
                "publishedAt": snippet.get("publishedAt"),
                "viewCount": stats.get("viewCount"),
                "likeCount": stats.get("likeCount"),
                "commentCount": int(stats.get("commentCount", 0))
            }

    return metadata


# ==============================
# BUSCAR VÍDEOS RECENTES
# ==============================

def get_valid_videos(uploads_playlist):

    valid_videos = []
    next_page = None

    while True:

        req = youtube.playlistItems().list(
            part="snippet",
            playlistId=uploads_playlist,
            maxResults=50,
            pageToken=next_page
        )

        res = req.execute()

        video_ids = []

        for item in res["items"]:
            video_ids.append(item["snippet"]["resourceId"]["videoId"])

        metadata = get_video_metadata(video_ids)

        for vid in video_ids:

            comment_count = metadata.get(vid, {}).get("commentCount", 0)

            if comment_count >= MAX_COMMENTS_PER_VIDEO:
                valid_videos.append(vid)

            if len(valid_videos) == VIDEOS_PER_CHANNEL:
                return valid_videos

        next_page = res.get("nextPageToken")

        if not next_page:
            break

    return valid_videos


# ==============================
# COLETA DE COMENTÁRIOS
# ==============================

def fetch_comments(video_id):

    comments = []
    next_page = None
    seen_ids = set()

    while len(comments) < MAX_COMMENTS_PER_VIDEO:

        try:

            req = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText",
                pageToken=next_page
            )

            res = req.execute()

        except HttpError as e:

            error_reason = ""

            try:
                error_reason = json.loads(
                    e.content.decode()
                )["error"]["errors"][0]["reason"]
            except:
                pass

            if error_reason == "commentsDisabled":
                print("Comentários desativados:", video_id)
                return []

            print("Erro inesperado:", e)
            return []

        for item in res["items"]:

            cid = item["snippet"]["topLevelComment"]["id"]

            if cid in seen_ids:
                continue

            seen_ids.add(cid)

            snippet = item["snippet"]["topLevelComment"]["snippet"]

            comments.append({
                "videoId": video_id,
                "commentId": cid,
                "publishedAt": snippet.get("publishedAt"),
                "text": snippet.get("textDisplay"),
                "likeCount": snippet.get("likeCount")
            })

            if len(comments) == MAX_COMMENTS_PER_VIDEO:
                break

        next_page = res.get("nextPageToken")

        if not next_page:
            break

        time.sleep(0.1)

    return comments


# ==============================
# SALVAR CSV
# ==============================

def save_csv(filename, data, fieldnames):

    filepath = os.path.join(os.getcwd(), filename)

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        if data:
            writer.writerows(data)

    print("Arquivo salvo em:", filepath)


# ==============================
# EXECUÇÃO PRINCIPAL
# ==============================

if __name__ == "__main__":

    print("Iniciando coleta")

    all_video_ids = []
    video_channel_type = {}

    for channel_name, channel_type in CHANNELS.items():

        print("Buscando canal:", channel_name)

        channel_id, uploads_playlist = get_channel_info(channel_name)

        if not channel_id:
            print("Canal não encontrado")
            continue

        videos = get_valid_videos(uploads_playlist)

        print(" -> vídeos válidos encontrados:", len(videos))

        if not videos:
            print("Nenhum vídeo válido encontrado")
            continue

        for v in videos:
            video_channel_type[v] = channel_type

        all_video_ids.extend(videos)

    print("Total de vídeos coletados:", len(all_video_ids))

    metadata = get_video_metadata(all_video_ids)

    all_comments = []

    for vid in all_video_ids:

        print("Coletando comentários:", metadata[vid]["title"])

        comments = fetch_comments(vid)

        if not comments:
            print("Nenhum comentário coletado")
            continue

        for c in comments:

            c["channel_type"] = video_channel_type.get(vid)
            c["video_title"] = metadata[vid]["title"]

        print(" ->", len(comments), "comentários")

        all_comments.extend(comments)

    print("Total de comentários coletados:", len(all_comments))

    save_csv(
        "raw_comments.csv",
        all_comments,
        [
            "videoId",
            "video_title",
            "commentId",
            "publishedAt",
            "text",
            "likeCount",
            "channel_type"
        ]
    )

    save_csv(
        "video_metadata.csv",
        list(metadata.values()),
        [
            "videoId",
            "title",
            "description",
            "channelId",
            "channelTitle",
            "publishedAt",
            "viewCount",
            "likeCount",
            "commentCount"
        ]
    )

    with open("raw_comments_backup.json", "w", encoding="utf-8") as f:
        json.dump(all_comments, f, ensure_ascii=False, indent=2)

    print("Coleta finalizada")

# ==============================
# BUSCAR CHANNEL INFO
# ==============================

def get_channel_info(channel_name):

    res = youtube.search().list(
        q=channel_name,
        part="snippet",
        type="channel",
        maxResults=1
    ).execute()

    items = res.get("items")

    if not items:
        return None, None

    channel_id = items[0]["snippet"]["channelId"]

    res = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()

    uploads_playlist = res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    return channel_id, uploads_playlist


# ==============================
# METADADOS DOS VÍDEOS
# ==============================

def get_video_metadata(video_ids):

    metadata = {}

    for i in range(0, len(video_ids), 50):

        block = video_ids[i:i+50]

        res = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(block)
        ).execute()

        for item in res["items"]:

            vid = item["id"]

            snippet = item["snippet"]
            stats = item["statistics"]

            metadata[vid] = {
                "videoId": vid,
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "channelId": snippet.get("channelId"),
                "channelTitle": snippet.get("channelTitle"),
                "publishedAt": snippet.get("publishedAt"),
                "viewCount": stats.get("viewCount"),
                "likeCount": stats.get("likeCount"),
                "commentCount": int(stats.get("commentCount", 0))
            }

    return metadata


# ==============================
# BUSCAR VÍDEOS RECENTES
# ==============================

def get_valid_videos(uploads_playlist):

    valid_videos = []
    next_page = None

    while True:

        req = youtube.playlistItems().list(
            part="snippet",
            playlistId=uploads_playlist,
            maxResults=50,
            pageToken=next_page
        )

        res = req.execute()

        video_ids = []

        for item in res["items"]:
            video_ids.append(item["snippet"]["resourceId"]["videoId"])

        metadata = get_video_metadata(video_ids)

        for vid in video_ids:

            comment_count = metadata.get(vid, {}).get("commentCount", 0)

            if comment_count >= MAX_COMMENTS_PER_VIDEO:
                valid_videos.append(vid)

            if len(valid_videos) == VIDEOS_PER_CHANNEL:
                return valid_videos

        next_page = res.get("nextPageToken")

        if not next_page:
            break

    return valid_videos


# ==============================
# COLETA DE COMENTÁRIOS
# ==============================

def fetch_comments(video_id):

    comments = []
    next_page = None
    seen_ids = set()

    while len(comments) < MAX_COMMENTS_PER_VIDEO:

        try:

            req = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText",
                pageToken=next_page
            )

            res = req.execute()

        except HttpError as e:

            error_reason = ""

            try:
                error_reason = json.loads(
                    e.content.decode()
                )["error"]["errors"][0]["reason"]
            except:
                pass

            if error_reason == "commentsDisabled":
                print("Comentários desativados:", video_id)
                return []

            print("Erro inesperado:", e)
            return []

        for item in res["items"]:

            cid = item["snippet"]["topLevelComment"]["id"]

            if cid in seen_ids:
                continue

            seen_ids.add(cid)

            snippet = item["snippet"]["topLevelComment"]["snippet"]

            comments.append({
                "videoId": video_id,
                "commentId": cid,
                "publishedAt": snippet.get("publishedAt"),
                "text": snippet.get("textDisplay"),
                "likeCount": snippet.get("likeCount")
            })

            if len(comments) == MAX_COMMENTS_PER_VIDEO:
                break

        next_page = res.get("nextPageToken")

        if not next_page:
            break

        time.sleep(0.1)

    return comments


# ==============================
# SALVAR CSV
# ==============================

def save_csv(filename, data, fieldnames):

    filepath = os.path.join(os.getcwd(), filename)

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        if data:
            writer.writerows(data)

    print("Arquivo salvo em:", filepath)


# ==============================
# EXECUÇÃO PRINCIPAL
# ==============================

if __name__ == "__main__":

    print("Iniciando coleta")

    all_video_ids = []
    video_channel_type = {}

    for channel_name, channel_type in CHANNELS.items():

        print("Buscando canal:", channel_name)

        channel_id, uploads_playlist = get_channel_info(channel_name)

        if not channel_id:
            print("Canal não encontrado")
            continue

        videos = get_valid_videos(uploads_playlist)

        print(" -> vídeos válidos encontrados:", len(videos))

        if not videos:
            print("Nenhum vídeo válido encontrado")
            continue

        for v in videos:
            video_channel_type[v] = channel_type

        all_video_ids.extend(videos)

    print("Total de vídeos coletados:", len(all_video_ids))

    metadata = get_video_metadata(all_video_ids)

    all_comments = []

    for vid in all_video_ids:

        print("Coletando comentários:", metadata[vid]["title"])

        comments = fetch_comments(vid)

        if not comments:
            print("Nenhum comentário coletado")
            continue

        for c in comments:

            c["channel_type"] = video_channel_type.get(vid)
            c["video_title"] = metadata[vid]["title"]

        print(" ->", len(comments), "comentários")

        all_comments.extend(comments)

    print("Total de comentários coletados:", len(all_comments))

    save_csv(
        "raw_comments.csv",
        all_comments,
        [
            "videoId",
            "video_title",
            "commentId",
            "publishedAt",
            "text",
            "likeCount",
            "channel_type"
        ]
    )

    save_csv(
        "video_metadata.csv",
        list(metadata.values()),
        [
            "videoId",
            "title",
            "description",
            "channelId",
            "channelTitle",
            "publishedAt",
            "viewCount",
            "likeCount",
            "commentCount"
        ]
    )

    with open("raw_comments_backup.json", "w", encoding="utf-8") as f:
        json.dump(all_comments, f, ensure_ascii=False, indent=2)

    print("Coleta finalizada")