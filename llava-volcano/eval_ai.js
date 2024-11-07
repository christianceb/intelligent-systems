/* JavaScript code to process tabulated_predict_results into Eval AI input */

const anno = []
const answers = []

for (const annotation of anno) {
  let findIndex = answers.findIndex(answer => answer.name == annotation.name)

  if (findIndex === -1) {
    findIndex = answers.push({
      name: annotation.name,
      attributes: []
    }) - 1
  }

  answers[findIndex].attributes.push({
    _key: annotation.key,
    answer: annotation.result === "Left" ? "After" : "Before"
  })
}

for (const annotation of answers) {
  for (let attribute of annotation.attributes) {
    if (attribute.key === undefined) {
      let count = 0;

      for (let kindOfAttribute of annotation.attributes) {
        if (kindOfAttribute._key == attribute._key) {
          kindOfAttribute['key'] = attribute._key + "_" + count
          count++;
        }
      }
    }
  }
}


for (const annotation of answers) {
  for (let attribute of annotation.attributes) {
    delete attribute._key;
  }
}

console.log(JSON.stringify(answers))