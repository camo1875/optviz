# 배포 가이드

이 프로젝트를 두 가지 방법으로 공개합니다.

| 대상 | 방법 | 결과 |
|---|---|---|
| 코드를 보는 사람 | GitHub 저장소 | `https://github.com/<아이디>/optimizer-visualize` |
| 애니메이션만 보는 사람 | GitHub Pages (같은 저장소의 `docs/` 폴더) | `https://<아이디>.github.io/optimizer-visualize/` |

웹 페이지는 계산을 모두 브라우저 안에서 하므로 서버가 필요 없고, GitHub Pages는 무료입니다.

---

## 0. 올리기 전에 정할 것

- **보고서 PDF는 올리지 않습니다**: 이름, 학교, 지도교사 이름이 들어 있어서 `.gitignore`에 넣어 두었습니다.
  `git status`로 PDF가 목록에 없는지 한 번 확인하세요.
- **커밋 작성자 이름과 이메일**: 커밋에 기록되어 공개됩니다. 현재 설정을 확인하고, 필요하면 이 저장소에서만 바꾸세요.
  ```bash
  git config user.name; git config user.email          # 확인
  git config user.name "표시할 이름"                    # 이 저장소에서만 변경 (git init 뒤에)
  git config user.email "<아이디>@users.noreply.github.com"   # 실제 이메일을 숨기려면
  ```
- **라이선스** (선택): 다른 사람이 코드를 가져다 써도 되는지 밝히려면 GitHub에서 저장소를 만들 때 "Add a license"에서 MIT를 고르면 됩니다.

## 1. GitHub 저장소 만들기

### 방법 A. GitHub CLI (터미널에서 한 번에)

```bash
brew install gh          # 처음 한 번
gh auth login            # 브라우저로 로그인 (GitHub.com → HTTPS → Login with a web browser)

cd "/Users/icarrna/Documents/AGSTA/Optimizer Visualize"
git init -b main
git add .
git status               # .venv 가 목록에 없는지 확인
git commit -m "보고서 코드 복원 및 개량판, 웹 시각화"
gh repo create optimizer-visualize --public --source . --push \
  --description "딥러닝 옵티마이저 비교 시각화 (고등학교 수학과제연구, 2024)"
```

### 방법 B. 웹사이트에서 만들고 push

1. https://github.com/new 에서 Repository name에 `optimizer-visualize`를 쓰고 **Public**을 선택합니다.
   README, .gitignore, license는 **추가하지 않고** (이미 있음) Create repository를 누릅니다.
2. 터미널에서:
   ```bash
   cd "/Users/icarrna/Documents/AGSTA/Optimizer Visualize"
   git init -b main
   git add .
   git commit -m "보고서 코드 복원 및 개량판, 웹 시각화"
   git remote add origin https://github.com/<아이디>/optimizer-visualize.git
   git push -u origin main
   ```
   비밀번호를 물으면 GitHub 비밀번호가 아니라 **Personal access token**을 넣어야 합니다
   (GitHub → Settings → Developer settings → Personal access tokens). 번거로우면 방법 A를 쓰세요.

## 2. GitHub Pages 켜기 (애니메이션 웹 페이지)

1. 저장소 페이지 → **Settings** → 왼쪽 **Pages**
2. Build and deployment → Source: **Deploy from a branch**
3. Branch: **main**, 폴더: **/docs** → Save
4. 1~2분 뒤 `https://<아이디>.github.io/optimizer-visualize/` 에서 열립니다.

특정 실험으로 바로 여는 링크도 쓸 수 있습니다.

| 링크 끝 | 실험 |
|---|---|
| `#report` | 보고서 4.6 그대로 |
| `#adam-099` | Adam β₁ = 0.99 |
| `#adam-lr` | β₁은 그대로, 학습률만 크게 |
| `#near-local` | 극솟값 근처에서 출발 |
| `#bowl` | 길쭉한 그릇 |
| `#saddle` | 안장점 탈출 |
| `#rosenbrock` | Rosenbrock 계곡 |

## 3. README에 링크 달기

Pages 주소가 생기면 `README.md` 맨 위 "웹에서 바로 보기" 줄의 `<아이디>`를 실제 아이디로 바꾸고 다시 올리세요.

```bash
git add README.md && git commit -m "웹 페이지 링크 추가" && git push
```

## 4. 웹 페이지를 고친 뒤 반영하기

웹 페이지의 원본은 `web/app.html` 하나입니다. `docs/index.html`은 거기서 만들어지는 파일이므로 직접 고치지 마세요.

```bash
# web/app.html 수정 후
python web/build_pages.py        # docs/index.html 다시 만들기
git add web docs && git commit -m "웹 페이지 수정" && git push
```

push하고 1~2분 뒤 Pages에 반영됩니다.

## 5. (선택) claude.ai 링크로 공유

같은 페이지가 claude.ai 아티팩트로도 올라가 있습니다. 처음에는 본인만 볼 수 있으므로, 공유하려면 페이지의 **Share** 메뉴에서 공개 범위를 바꾸세요.
GitHub Pages 주소가 생기면 그쪽이 로그인 없이 누구나 열 수 있어 더 편합니다.
