/**
 * FFmpeg 자동 다운로드 스크립트
 * 빌드 전에 실행되어 Windows/Mac/Linux용 FFmpeg 바이너리를 다운로드합니다.
 */

const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const FFMPEG_DIR = path.join(__dirname, '..', 'ffmpeg');

// FFmpeg 다운로드 URL
const FFMPEG_URLS = {
  win: 'https://github.com/GyanD/codexffmpeg/releases/download/7.0.2/ffmpeg-7.0.2-essentials_build.zip',
  mac: 'https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip',
  linux: 'https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz'
};

function createDirectories() {
  console.log('📁 FFmpeg 디렉토리 생성 중...');
  
  ['win', 'mac', 'linux'].forEach(platform => {
    const dir = path.join(FFMPEG_DIR, platform);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
      console.log(`   ✅ ${platform} 디렉토리 생성 완료`);
    }
  });
}

function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : http;
    const file = fs.createWriteStream(dest);
    
    console.log(`   📥 다운로드 중: ${url}`);
    
    protocol.get(url, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        // 리다이렉트 처리
        file.close();
        fs.unlinkSync(dest);
        return downloadFile(response.headers.location, dest)
          .then(resolve)
          .catch(reject);
      }
      
      if (response.statusCode !== 200) {
        reject(new Error(`다운로드 실패: ${response.statusCode}`));
        return;
      }
      
      const totalSize = parseInt(response.headers['content-length'], 10);
      let downloadedSize = 0;
      
      response.on('data', (chunk) => {
        downloadedSize += chunk.length;
        const percent = ((downloadedSize / totalSize) * 100).toFixed(1);
        process.stdout.write(`\r   진행률: ${percent}%`);
      });
      
      response.pipe(file);
      
      file.on('finish', () => {
        file.close();
        console.log('\n   ✅ 다운로드 완료');
        resolve();
      });
    }).on('error', (err) => {
      fs.unlinkSync(dest);
      reject(err);
    });
  });
}

function extractZip(zipPath, destDir) {
  console.log(`   📦 압축 해제 중: ${zipPath}`);
  
  try {
    if (process.platform === 'win32') {
      // Windows: PowerShell 사용
      execSync(`powershell -command "Expand-Archive -Path '${zipPath}' -DestinationPath '${destDir}' -Force"`, {
        stdio: 'inherit'
      });
    } else {
      // Mac/Linux: unzip 사용
      execSync(`unzip -o "${zipPath}" -d "${destDir}"`, {
        stdio: 'inherit'
      });
    }
    console.log('   ✅ 압축 해제 완료');
  } catch (error) {
    console.error('   ❌ 압축 해제 실패:', error.message);
    throw error;
  }
}

function extractTarXz(tarPath, destDir) {
  console.log(`   📦 압축 해제 중: ${tarPath}`);
  
  try {
    execSync(`tar -xf "${tarPath}" -C "${destDir}"`, {
      stdio: 'inherit'
    });
    console.log('   ✅ 압축 해제 완료');
  } catch (error) {
    console.error('   ❌ 압축 해제 실패:', error.message);
    throw error;
  }
}

function moveFFmpegBinaries(extractDir, platform) {
  console.log(`   📋 바이너리 파일 이동 중...`);
  
  const destDir = path.join(FFMPEG_DIR, platform);
  
  // 추출된 디렉토리에서 ffmpeg 찾기
  function findFFmpeg(dir) {
    const items = fs.readdirSync(dir);
    
    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stat = fs.statSync(fullPath);
      
      if (stat.isDirectory()) {
        const result = findFFmpeg(fullPath);
        if (result) return result;
      } else if (item === 'ffmpeg.exe' || item === 'ffmpeg') {
        return dir;
      }
    }
    return null;
  }
  
  const binDir = findFFmpeg(extractDir);
  
  if (binDir) {
    const files = fs.readdirSync(binDir);
    
    files.forEach(file => {
      if (file.startsWith('ffmpeg') || file.startsWith('ffprobe')) {
        const src = path.join(binDir, file);
        const dest = path.join(destDir, file);
        
        fs.copyFileSync(src, dest);
        
        // Mac/Linux: 실행 권한 부여
        if (platform !== 'win') {
          fs.chmodSync(dest, 0o755);
        }
        
        console.log(`   ✅ ${file} 이동 완료`);
      }
    });
  } else {
    console.error('   ❌ FFmpeg 바이너리를 찾을 수 없습니다.');
  }
}

async function downloadFFmpegForPlatform(platform) {
  console.log(`\n🔽 ${platform.toUpperCase()} FFmpeg 다운로드 시작...`);
  
  const url = FFMPEG_URLS[platform];
  const tempDir = path.join(__dirname, '..', 'temp');
  
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }
  
  const fileName = url.split('/').pop();
  const filePath = path.join(tempDir, fileName);
  
  try {
    // 다운로드
    await downloadFile(url, filePath);
    
    // 압축 해제
    const extractDir = path.join(tempDir, platform);
    if (!fs.existsSync(extractDir)) {
      fs.mkdirSync(extractDir, { recursive: true });
    }
    
    if (filePath.endsWith('.zip')) {
      extractZip(filePath, extractDir);
    } else if (filePath.endsWith('.tar.xz')) {
      extractTarXz(filePath, extractDir);
    }
    
    // 바이너리 파일 이동
    moveFFmpegBinaries(extractDir, platform);
    
    // 임시 파일 삭제
    console.log('   🗑️  임시 파일 정리 중...');
    fs.rmSync(tempDir, { recursive: true, force: true });
    
    console.log(`✅ ${platform.toUpperCase()} FFmpeg 설치 완료!\n`);
    
  } catch (error) {
    console.error(`❌ ${platform.toUpperCase()} FFmpeg 다운로드 실패:`, error.message);
  }
}

async function checkExistingFFmpeg() {
  console.log('🔍 기존 FFmpeg 확인 중...\n');
  
  const platforms = ['win', 'mac', 'linux'];
  const existing = [];
  
  platforms.forEach(platform => {
    const ffmpegPath = path.join(FFMPEG_DIR, platform, platform === 'win' ? 'ffmpeg.exe' : 'ffmpeg');
    if (fs.existsSync(ffmpegPath)) {
      existing.push(platform);
      console.log(`   ✅ ${platform.toUpperCase()} FFmpeg 이미 존재`);
    } else {
      console.log(`   ❌ ${platform.toUpperCase()} FFmpeg 없음`);
    }
  });
  
  return existing;
}

async function main() {
  console.log('╔════════════════════════════════════════╗');
  console.log('║   FFmpeg 자동 다운로드 스크립트      ║');
  console.log('╚════════════════════════════════════════╝\n');
  
  // 디렉토리 생성
  createDirectories();
  
  // 기존 FFmpeg 확인
  const existing = await checkExistingFFmpeg();
  
  // 필요한 플랫폼만 다운로드
  const platforms = ['win', 'mac', 'linux'];
  const toDownload = platforms.filter(p => !existing.includes(p));
  
  if (toDownload.length === 0) {
    console.log('\n✅ 모든 플랫폼의 FFmpeg가 이미 존재합니다!');
    console.log('   다시 다운로드하려면 ffmpeg 폴더를 삭제하세요.\n');
    return;
  }
  
  console.log(`\n📥 ${toDownload.length}개 플랫폼 다운로드 시작...\n`);
  
  // 순차적으로 다운로드 (병렬은 네트워크 부담)
  for (const platform of toDownload) {
    await downloadFFmpegForPlatform(platform);
  }
  
  console.log('╔════════════════════════════════════════╗');
  console.log('║   FFmpeg 다운로드 완료! 🎉           ║');
  console.log('╚════════════════════════════════════════╝');
}

// 실행
main().catch(error => {
  console.error('\n❌ 치명적 오류:', error);
  process.exit(1);
});