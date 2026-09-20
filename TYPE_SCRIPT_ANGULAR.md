Original goal/question: 
- Understand what Observable and Promise are in Angular/TypeScript
- Understand what "lazy" and "eager" mean in the context of Observables and Promises
- Understand what the "export" and "get" keywords do in TypeScript
- Understand if users can modify TypeScript logic in an Angular app from the browser

Key points:
- Observables handle asynchronous data streams, emit multiple values over time, and are lazy (only execute on subscription)
- Promises handle single asynchronous operations, emit one value, and are eager (execute immediately)
- The "export" keyword makes variables, functions, classes, etc. from a file accessible to other files that import them
- Default exports are the main export of a file, while named exports allow exporting multiple things 
- The "get" keyword defines a getter that looks like a property but runs code when accessed
- Getters are commonly used for computed properties, protecting private data, and validation
- Users cannot directly modify TypeScript logic in an Angular app because:
  - TypeScript is compiled to JavaScript before reaching the browser
  - Angular removes elements hidden by *ngIf from the DOM
  - Modifying JS in the browser only affects that user's local instance
- To fully hide sensitive logic from users, you can move it to the backend, obfuscate the JS, or use server-side rendering

Open questions / unresolved points:
- N/A

Current state:
- All original questions have been addressed and explained in detail
- The conversation could be considered complete unless you have any other related questions