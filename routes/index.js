var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});

router.get('/train', function(req, res, next) {
  res.render('train');
});

router.get('/predict', function(req, res, next) {
  res.render('predict');
});

router.get('/store/:ff', function(req, res, next) {
  res.download('./public/model/'+req.params.ff)
});

router.post('/model', function(req, res, next) {
  console.log(req.fields); // contains non-file fields
  console.log(req.files); // contains files
  res.status(200).end('Sucess')
});

module.exports = router;
