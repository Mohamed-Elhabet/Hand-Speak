const allSideMenu = document.querySelectorAll('#sidebar .side-menu.top li a');
// allSideMenu.forEach(item=> {
//     const li = item.parentElement;
//     item.addEventListener('click', function () {
//         allSideMenu.forEach(i=> {
//             i.parentElement.classList.remove('active');
//         })
//         li.classList.add('active');
//     })
// });

// TOGGLE SIDEBAR
const menuBar = document.querySelector('#content nav .bx.bx-menu');
const sidebar = document.getElementById('sidebar');
menuBar.addEventListener('click', function () {
    sidebar.classList.toggle('hide');
})
const searchButton = document.querySelector('#content nav form .form-input button');
const searchButtonIcon = document.querySelector('#content nav form .form-input button .bx');
const searchForm = document.querySelector('#content nav form');
searchButton.addEventListener('click', function (e) {
    if(window.innerWidth < 576) {
        e.preventDefault();
        searchForm.classList.toggle('show');
        if(searchForm.classList.contains('show')) {
            searchButtonIcon.classList.replace('bx-search', 'bx-x');
        } else {
            searchButtonIcon.classList.replace('bx-x', 'bx-search');
        }
    }
})
if(window.innerWidth < 768) {
    sidebar.classList.add('hide');
} else if(window.innerWidth > 576) {
    searchButtonIcon.classList.replace('bx-x', 'bx-search');
    searchForm.classList.remove('show');
}
window.addEventListener('resize', function () {
    if(this.innerWidth > 576) {
        searchButtonIcon.classList.replace('bx-x', 'bx-search');
        searchForm.classList.remove('show');
    }
})



const switchMode = document.getElementById('switch-mode');
const storeMode = localStorage.getItem('mode')

if(storeMode === 'dark'){
    switchMode.checked = true;
    document.body.classList.add('dark')
}else{
    switchMode.checked = false;
    document.body.classList.remove('dark')
}

switchMode.addEventListener('change', function () {
    if(this.checked) {
        document.body.classList.add('dark');
        localStorage.setItem('mode', 'dark')
    } else {
        document.body.classList.remove('dark');
        localStorage.setItem('mode', 'light')
    }
})



// Speak text 

// function speakText() {
//     var textElement = document.getElementById('translatedText');
//     var translatedText = textElement.innerText.trim();
//     var ttsUrl = 'https://translate.google.com/translate_tts?ie=UTF-8&q=' + encodeURIComponent(translatedText) + '&tl=en&total=1&idx=0&textlen=' + translatedText.length + '&client=tw-ob&prev=input';
//     var audioElement = new Audio(ttsUrl);
//     audioElement.play();
// }


function speakText(translatedText) {
    var ttsUrl = 'https://translate.google.com/translate_tts?ie=UTF-8&q=' + encodeURIComponent(translatedText) + '&tl=en&total=1&idx=0&textlen=' + translatedText.length + '&client=tw-ob&prev=input';
    var audioElement = new Audio(ttsUrl);
    audioElement.play();
}
