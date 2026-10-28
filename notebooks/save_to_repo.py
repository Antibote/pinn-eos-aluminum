# Отправка папки artifacts на GitHub из Colab. Запускается из раздела 0 после ячейки с репозиторием:
#   %run -i pinn-eos-aluminum/notebooks/save_to_repo.py
# Нужны REPO, REPO_URL, ARTIFACTS, git() из ячейки «Репозиторий». Без секрета GITHUB_TOKEN
# ничего не отправляет: результаты остаются в artifacts/ текущей среды Colab.


def save_to_repo(message):
    """Коммитит папку artifacts и отправляет её на GitHub.

    Нужен секрет Colab GITHUB_TOKEN (значок ключа слева) с правом записи в репозиторий.
    Без него функция только сообщает, где лежат файлы, — ячейки ноутбука не падают.

    Args:
        message: что сохранили, например 'Бейзлайн: метрики и карта ошибок'.
    """
    global _TOKEN
    try:
        from google.colab import userdata
        _TOKEN = userdata.get('GITHUB_TOKEN')
    except Exception:  # не Colab, секрета нет или нет доступа к нему
        _TOKEN = None
    if not _TOKEN:
        print(f'GITHUB_TOKEN не задан — результаты сохранены только в {ARTIFACTS.resolve()} (на GitHub не отправлены)')
        return

    big = [str(p) for p in ARTIFACTS.rglob('*') if p.is_file() and p.stat().st_size > 50 * 2**20]
    if big:
        raise RuntimeError(f'Файлы больше 50 МБ GitHub не примет: {big}')
    git('add', 'artifacts')
    changed = git('diff', '--cached', '--name-only', '--', 'artifacts').split()
    if not changed:
        print('Нечего сохранять: artifacts не изменилась')
        return

    # Автор коммита — владелец токена
    req = urllib.request.Request('https://api.github.com/user', headers={'Authorization': f'token {_TOKEN}'})
    me = json.load(urllib.request.urlopen(req))
    author = f"{me.get('name') or me['login']} <{me['id']}+{me['login']}@users.noreply.github.com>"

    git('commit', '-q', '-m', message, '--author', author)
    url = REPO_URL.replace('https://', f'https://x-access-token:{_TOKEN}@')
    git('pull', '-q', '--rebase', url, 'main')
    git('push', '-q', url, 'HEAD:main')
    print('На GitHub:', git('log', '-1', '--format=%h %s').strip())
    print('Файлы:', *changed, sep='\n  ')
